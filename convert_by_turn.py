import json
import os
import re
from copy import deepcopy

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

data_dir = "data/mwz2.1"

data_files = ["train_dials_v2.json", "dev_dials_v2.json", "test_dials_v2.json"]

#--------------------------------

def make_slot_meta(ontology):
    meta = []
    change = {}
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if d not in EXPERIMENT_DOMAINS:
            continue
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change


### Read ontology file
fp_ont = open(os.path.join(data_dir, "ontology-modified_v2.json"), "r")
data_ont = json.load(fp_ont)
fp_ont.close()

slot_meta, _ = make_slot_meta(data_ont)
ontology_modified = {}
for slot in slot_meta:
    ontology_modified[slot] = []

### Read data and write to tsv files
fp_train = open(os.path.join(data_dir, "train.tsv"), "w")
fp_dev = open(os.path.join(data_dir, "dev.tsv"), "w")
fp_test = open(os.path.join(data_dir, "test.tsv"), "w")

fp_train.write('# Dialogue ID\tTurn Index\tLast Turn\tSystem Response\tUser Utterance\t')
fp_dev.write('# Dialogue ID\tTurn Index\tLast Turn\tSystem Response\tUser Utterance\t')
fp_test.write('# Dialogue ID\tTurn Index\tLast Turn\tSystem Response\tUser Utterance\t')

for slot in slot_meta:
    fp_train.write(str(slot) + '\t')
    fp_dev.write(str(slot) + '\t')
    fp_test.write(str(slot) + '\t')
    
fp_train.write('\n')
fp_dev.write('\n')
fp_test.write('\n')


for idx, file_id in enumerate(data_files):
    if idx==0:
        fp_out = fp_train
    elif idx==1:
        fp_out = fp_dev
    else:
        fp_out = fp_test
        
    fp_data = open(os.path.join(data_dir, file_id), "r")
    dials = json.load(fp_data)
    
    for dial_dict in dials:
        tidx = 0
        prev_belief = {}
        for slot in slot_meta:
            prev_belief[slot] = "none"
        dial_domains = dial_dict["domains"]
        for ti, turn in enumerate(dial_dict["dialogue"]):
            turn_domain = turn["domain"]
            if turn_domain not in EXPERIMENT_DOMAINS:
                continue
                
            fp_out.write(str(dial_dict["dialogue_idx"])) # dialogue id
            fp_out.write('\t' + str(tidx)) # turn id
            tidx += 1
            if (ti + 1) == len(dial_dict["dialogue"]): # last turn
                fp_out.write('\t' + str("True"))
            else:
                fp_out.write('\t' + str("False"))
            fp_out.write('\t' + str(turn["system_transcript"])) # system response
            fp_out.write('\t' + str(turn["transcript"])) # user utterance
                       
            turn_dialog_state = deepcopy(prev_belief)
            for tl in turn["turn_label"]:
                turn_dialog_state[tl[0]] = tl[1]

            for slot in slot_meta:
                fp_out.write('\t' + turn_dialog_state[slot])
                ontology_modified[slot].append(turn_dialog_state[slot])
            
            prev_belief = turn_dialog_state
            
            fp_out.write('\n')
            fp_out.flush()

fp_train.close()
fp_dev.close()
fp_test.close()

for slot in slot_meta:
    ontology_modified[slot] = sorted(list(set(ontology_modified[slot])))
    
with open(os.path.join(data_dir, 'ontology-modified.json'), 'w') as outfile:
     json.dump(ontology_modified, outfile, indent=4)