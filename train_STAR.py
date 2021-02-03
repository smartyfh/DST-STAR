# import faulthandler
# faulthandler.enable()
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import json
import time
import logging
from tqdm import tqdm, trange

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from utils.data_utils import Processor, MultiWozDataset
from utils.eval_utils import model_evaluation
from utils.label_lookup import get_label_lookup_from_first_token, combine_slot_values
from models.ModelBERT import UtteranceEncoding
from models.ModelBERT import BeliefTracker

from transformers import BertTokenizer
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.cuda.set_device(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # logger
    logger_file_name = args.save_dir.split('/')[1]
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "%s.txt"%(logger_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)
    
    # cuda setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: {}".format(device))
    
    # set random seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #******************************************************
    # load data
    #******************************************************
    processor = Processor(args)
    slot_meta = processor.slot_meta
    label_list = processor.label_list
    num_labels = [len(labels) for labels in label_list]
    logger.info(slot_meta)
   
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    
    train_data_raw = processor.get_train_instances(args.data_dir, tokenizer)
    print("# train examples %d" % len(train_data_raw))
    
    dev_data_raw = processor.get_dev_instances(args.data_dir, tokenizer)
    print("# dev examples %d" % len(dev_data_raw))
    
    test_data_raw = processor.get_test_instances(args.data_dir, tokenizer)
    print("# test examples %d" % len(test_data_raw))
    logger.info("Data loaded!")
    
    train_data = MultiWozDataset(train_data_raw,
                                 tokenizer,
                                 word_dropout=args.word_dropout)
    
    num_train_steps = int(len(train_data_raw) / args.train_batch_size * args.n_epochs)
    logger.info("***** Run training *****")
    logger.info(" Num examples = %d", len(train_data_raw))
    logger.info(" Batch size = %d", args.train_batch_size)
    logger.info(" Num steps = %d", num_train_steps)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=train_data.collate_fn,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn)

    #******************************************************
    # build model
    #******************************************************
    ## Initialize slot and value embeddings
    sv_encoder = UtteranceEncoding.from_pretrained(args.pretrained_model)
    for p in sv_encoder.bert.parameters():
        p.requires_grad = False
    
    new_label_list, slot_value_pos = combine_slot_values(slot_meta, label_list) # without slot head
    logger.info(slot_value_pos)
    slot_lookup = get_label_lookup_from_first_token(slot_meta, tokenizer, sv_encoder, device)
    value_lookup = get_label_lookup_from_first_token(new_label_list, tokenizer, sv_encoder, device)
    
    model = BeliefTracker(args, slot_lookup, value_lookup, num_labels, slot_value_pos, device)
    model.to(device)
   
    ## prepare optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.enc_lr)
    enc_scheduler = get_linear_schedule_with_warmup(enc_optimizer, int(num_train_steps * args.enc_warmup), num_train_steps)

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    dec_scheduler = get_linear_schedule_with_warmup(dec_optimizer, int(num_train_steps * args.dec_warmup), num_train_steps)
    
    logger.info(enc_optimizer)
    logger.info(dec_optimizer)

    #******************************************************
    # training
    #******************************************************
    logger.info("Training...")
    
    best_loss = None
    best_acc = None
    last_update = None
    for epoch in trange(int(args.n_epochs), desc="Epoch"):
        batch_loss = []
        batch_acc = []

        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()

            batch = [b.to(device) if b is not None else b for b in batch]
            input_ids, segment_ids, input_mask, label_ids = batch

            # forward
            loss, _, acc, _, _ = model(input_ids=input_ids, attention_mask=input_mask,
                                       token_type_ids=segment_ids, labels=label_ids)

            loss.backward()
            enc_optimizer.step()
            enc_scheduler.step()
            dec_optimizer.step()
            dec_scheduler.step()
            model.zero_grad()
            
            batch_loss.append(loss.item())
            batch_acc.append(acc)
            if step % 300 == 0:
                print("[%d/%d] [%d/%d] mean_loss: %.6f, mean_joint_acc: %.6f" % \
                      (epoch+1, args.n_epochs, step, len(train_dataloader), np.mean(batch_loss), np.mean(batch_acc)))
                batch_loss = []
                batch_acc = []
                
            if epoch > args.n_epochs / 2 and step > 0 and step % args.eval_step == 0:
                eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, label_list, epoch*10+step/args.eval_step)
                if last_update is None or best_loss > eval_res['loss']:
                    best_loss = eval_res['loss']
                    save_path = os.path.join(args.save_dir, 'model_best_loss.bin')
                    torch.save(model.state_dict(), save_path)
                    print("Best Loss : ", best_loss)
                    print("\n")
                if last_update is None or best_acc < eval_res['joint_acc']:
                    best_acc = eval_res['joint_acc']
                    save_path = os.path.join(args.save_dir, 'model_best_acc.bin')
                    torch.save(model.state_dict(), save_path)
                    print("Best Acc : ", best_acc)
                    print("\n")

                logger.info("*** Step=%d, Dev Loss=%.6f, Dev Acc=%.6f, Dev Turn Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % \
                           (step, eval_res['loss'], eval_res['joint_acc'], eval_res['joint_turn_acc'], best_loss, best_acc))
                
            if epoch > args.n_epochs / 2 and step > 0 and step % args.eval_step == 0:
                eval_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, \
                                            label_list, epoch*10+step/args.eval_step)

                logger.info("*** Step=%d, Tes Loss=%.6f, Tes Acc=%.6f, Tes Turn Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % \
                           (step, eval_res['loss'], eval_res['joint_acc'], eval_res['joint_turn_acc'], best_loss, best_acc))

        if (epoch+1) % args.eval_epoch == 0:
            eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, label_list, epoch+1)
            if last_update is None or best_loss > eval_res['loss']:
                best_loss = eval_res['loss']
                save_path = os.path.join(args.save_dir, 'model_best_loss.bin')
                torch.save(model.state_dict(), save_path)
                print("Best Loss : ", best_loss)
                print("\n")
            if last_update is None or best_acc < eval_res['joint_acc']:
                best_acc = eval_res['joint_acc']
                save_path = os.path.join(args.save_dir, 'model_best_acc.bin')
                torch.save(model.state_dict(), save_path)
                last_update = epoch
                print("Best Acc : ", best_acc)
                print("\n")
               
            logger.info("*** Epoch=%d, Last Update=%d, Dev Loss=%.6f, Dev Acc=%.6f, Dev Turn Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % (epoch, last_update, eval_res['loss'], eval_res['joint_acc'], eval_res['joint_turn_acc'], best_loss, best_acc))
            
            
        if (epoch+1) % args.eval_epoch == 0:
            eval_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, label_list, epoch+1)
               
            logger.info("*** Epoch=%d, Last Update=%d, Tes Loss=%.6f, Tes Acc=%.6f, Tes Turn Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % (epoch, last_update, eval_res['loss'], eval_res['joint_acc'], eval_res['joint_turn_acc'], best_loss, best_acc))
            
        if last_update + args.patience <= epoch:
                break


    print("Test using best loss model...")
    best_epoch = 0
    ckpt_path = os.path.join(args.save_dir, 'model_best_loss.bin')
    model = BeliefTracker(args, slot_lookup, value_lookup, num_labels, slot_value_pos, device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    test_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, label_list, 
                                best_epoch, is_gt_p_state=False)
    logger.info("Results based on best loss: ")
    logger.info(test_res)
    #----------------------------------------------------------------------
    print("Test using best acc model...")
    ckpt_path = os.path.join(args.save_dir, 'model_best_acc.bin')
    model = BeliefTracker(args, slot_lookup, value_lookup, num_labels, slot_value_pos, device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    test_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, label_list, 
                                best_epoch+1, is_gt_p_state=False)
    logger.info("Results based on best acc: ")
    logger.info(test_res)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='data/mwz2.1', type=str)
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str)
    parser.add_argument("--save_dir", default='out-bert/exp', type=str)
    parser.add_argument("--attn_type", default='softmax', type=str,
                        help="softmax or tanh")

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=12, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--eval_step", default=10000, type=int,
                        help="Within each epoch, do evaluation as well at every eval_step")

    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--attn_head", default=4, type=int)
    parser.add_argument("--num_history", default=20, type=int)
    parser.add_argument("--distance_metric", default="euclidean", type=str,
                        help="euclidean or cosine")
    
    parser.add_argument("--num_self_attention_layer", default=6, type=int)
    
    args = parser.parse_args()
    
    print('pytorch version: ', torch.__version__)
#     print(args)
    main(args)
