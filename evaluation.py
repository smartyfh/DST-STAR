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
from utils.generate_full_prediction import model_evaluation
from utils.label_lookup import get_label_lookup_from_first_token, combine_slot_values
from models.ModelBERT import UtteranceEncoding
from models.ModelBERT import BeliefTracker

from transformers import BertTokenizer


# os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.cuda.set_device(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # logger
    logger_file_name = args.save_dir.split('/')[1]
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "%s.txt"%(logger_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)
    
    # cuda setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: {}".format(device))

    #******************************************************
    # load data
    #******************************************************
    processor = Processor(args)
    slot_meta = processor.slot_meta
    label_list = processor.label_list
    num_labels = [len(labels) for labels in label_list]
    logger.info(slot_meta)
   
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    
    test_data_raw = processor.get_test_instances(args.data_dir, tokenizer)
    print("# test examples %d" % len(test_data_raw))
    logger.info("Data loaded!")
    
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
    
    # load state_dict
    ckpt_path = os.path.join(args.save_dir, 'model_best_acc.bin')
    model = BeliefTracker(args, slot_lookup, value_lookup, num_labels, slot_value_pos, device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    
    best_epoch = 0
    test_res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, label_list, 
                                best_epoch, is_gt_p_state=False)
    logger.info("Results based on best acc: ")
    logger.info(test_res)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='data/mwz2.1', type=str)
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str)
    parser.add_argument("--save_dir", default='out-bert/exp', type=str)
    parser.add_argument("--attn_type", default='softmax', type=str)
    parser.add_argument("--dropout_prob", default=0.1, type=float)

    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--attn_head", default=4, type=int)
    parser.add_argument("--num_history", default=20, type=int)
    parser.add_argument("--distance_metric", default="euclidean", type=str)
    
    parser.add_argument("--num_self_attention_layer", default=6, type=int)
    
    args = parser.parse_args()
    
    print('pytorch version: ', torch.__version__)
    print(args)
    main(args)
