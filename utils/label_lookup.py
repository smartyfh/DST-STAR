import torch
import torch.nn as nn
from .data_utils import slot_recovery


def combine_slot_values(slot_meta, label_list, add_slot_head=False): # flatten
    new_label_list = []
    slot_value_pos = []
    st = 0
    ed = 0
    for s, slot in enumerate(slot_meta):
        slot_labels = label_list[s]
        ed += len(slot_labels)
        slot_value_pos.append([st, ed])
        st = ed
        slot = slot_recovery(slot)
        for label in slot_labels:
            if add_slot_head:
                new_label_list.append(slot + " " + label)
            else:
                new_label_list.append(label)
    
    return new_label_list, slot_value_pos

    
def get_label_ids(labels, tokenizer):
    label_ids = []
    label_lens = []
    max_len = 0
    for label in labels:
        label_token_ids = tokenizer(label)["input_ids"] # special tokens added automatically
        label_len = len(label_token_ids)
        max_len = max(max_len, label_len)
  
        label_ids.append(label_token_ids)
        label_lens.append(label_len)
    
    label_ids_padded = []
    for label_item_ids in label_ids:
        item_len = len(label_item_ids)
        padding = [0] * (max_len - item_len)
        label_ids_padded.append(label_item_ids + padding)
    label_ids_padded = torch.tensor(label_ids_padded, dtype=torch.long)

    return label_ids_padded, label_lens    


def get_label_lookup(labels, tokenizer, sv_encoder, device, use_layernorm=True):
    model_output_dim = sv_encoder.config.hidden_size
    label_lookup = nn.Embedding(len(labels), model_output_dim)
    
    sv_encoder.eval() 
    LN = nn.LayerNorm(model_output_dim, elementwise_affine=False)
    
    # get label ids
    label_ids, label_lens = get_label_ids(labels, tokenizer)

    # encoding
    label_type_ids = torch.zeros(label_ids.size(), dtype=torch.long)
    label_mask = label_ids > 0
    hid_label = sv_encoder(label_ids, label_mask, label_type_ids)[0]
    hid_label = hid_label.detach()
    # remove [CLS] and [SEP]
    for lb in range(label_ids.size(0)):
        label_mask[lb, 0] = False
        label_mask[lb, label_lens[lb]-1] = False
    expanded_label_mask = label_mask.view(-1, label_ids.size(1), 1).expand(hid_label.size()).float()
    hid_label = torch.mul(hid_label, expanded_label_mask)
    masked_label_len = torch.sum(expanded_label_mask, 1, True).repeat(1, label_ids.size(1), 1)
    hid_label = torch.mean(torch.true_divide(hid_label, masked_label_len), 1)
    if use_layernorm:
        hid_label = LN(hid_label)
    label_lookup = nn.Embedding.from_pretrained(hid_label, freeze=True).to(device)

    return label_lookup


def get_label_lookup_from_first_token(labels, tokenizer, sv_encoder, device, use_layernorm=False):
    model_output_dim = sv_encoder.config.hidden_size
    label_lookup = nn.Embedding(len(labels), model_output_dim)
    
    sv_encoder.eval() 
    LN = nn.LayerNorm(model_output_dim, elementwise_affine=False)
    
    # get label ids
    label_ids, label_lens = get_label_ids(labels, tokenizer)

    # encoding
    label_type_ids = torch.zeros(label_ids.size(), dtype=torch.long)
    label_mask = (label_ids > 0)
    hid_label = sv_encoder(label_ids, label_mask, label_type_ids)[0]
    hid_label = hid_label[:, 0, :]
    hid_label = hid_label.detach()
    if use_layernorm:
        hid_label = LN(hid_label)
    label_lookup = nn.Embedding.from_pretrained(hid_label, freeze=True).to(device)

    return label_lookup