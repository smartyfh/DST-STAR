import json
import os
import re

from utils.fix_label import fix_general_label_error


data_dir = "data/mwz2.1"

data_files = ["train_dials.json", "dev_dials.json", "test_dials.json"]

#--------------------------------
def normalize_time(text):
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text) # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text) # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text) # Missing separator
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text) # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text) # normalize simple full hour time
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text) # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text) # Correct times that use 24 as hour
    return text

def normalize_text(text):
    text = normalize_time(text)
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text) # Systematic typo
    text = re.sub("anthropogy", "anthropology", text)
    text = re.sub("theater", "theatre", text)
    text = re.sub("musuem", "museum", text)
    text = re.sub("the weat", "the west", text)
    text = re.sub("the wast", "the west", text)
    text = re.sub(" wendesday ", " wednesday ", text)
    text = re.sub(" wednes ", " wednesday ", text)
    text = re.sub("thurtsday", "thursday", text)
    text = re.sub("mdoerate", "moderate", text)
    text = re.sub("portugese", "portuguese", text)
    text = re.sub("guesthouse", "guest house", text) # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text) # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text) # Normalization
    return text

def normalize_label(slot, value_label):
    # Some general typos
    value_label = re.sub("theater", "theatre", value_label)
    value_label = re.sub("archaelogy", "archaeology", value_label)
    value_label = re.sub("anthropogy", "anthropology", value_label)
    value_label = re.sub("portugese", "portuguese", value_label)
    
    # Normalization of empty slots
    if value_label == '' or value_label == "not mentioned":
        return "none"

    # Normalization of time slots
    if "leaveat" in slot or "arriveby" in slot or slot == 'restaurant-book time':
        return normalize_time(value_label)

    # Normalization
    if "type" in slot or "name" in slot or "destination" in slot or "departure" in slot:
        value_label = re.sub("guesthouse", "guest house", value_label)
        
    if slot == "hotel-name":
        value_label = re.sub("b & b", "bed and breakfast", value_label)

    # Map to boolean slots
    if slot == 'hotel-parking' or slot == 'hotel-internet':
        if value_label == 'yes' or value_label == 'free':
            return "true"
        if value_label == "no":
            return "false"
    if slot == 'hotel-type':
        if value_label == "hotel":
            return "true"
        if value_label == "guest house":
            return "false"

    return value_label
#--------------------------------

def make_slot_meta(ontology):
    meta = []
    change = {}
    for i, k in enumerate(ontology.keys()):
        d, s = k.split('-')
        if 'price' in s or 'leave' in s or 'arrive' in s:
            s = s.replace(' ', '')
        meta.append('-'.join([d, s]))
        change[meta[-1]] = ontology[k]
    return sorted(meta), change


### Read ontology file
fp_ont = open(os.path.join(data_dir, "ontology.json"), "r")
data_ont = json.load(fp_ont)
fp_ont.close()

slot_meta, _ = make_slot_meta(data_ont)
ontology_modified = {}
for slot in slot_meta:
    ontology_modified[slot] = []

    
### normalize text and fix label errors
for idx, file_id in enumerate(data_files):   
    fp_data = open(os.path.join(data_dir, file_id), "r")
    dials = json.load(fp_data)
    
    dials_v2 = []
    for dial_dict in dials:
        dial_domains = dial_dict["domains"]
        prev_turn_state = {}
        for slot in slot_meta:
            prev_turn_state[slot] = "none"
            
        for ti, turn in enumerate(dial_dict["dialogue"]):             
            dial_dict["dialogue"][ti]["system_transcript"] = normalize_text(turn["system_transcript"])
            dial_dict["dialogue"][ti]["transcript"] = normalize_text(turn["transcript"])
            
            # state
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, slot_meta)
            for slot in slot_meta:
                if slot not in turn_dialog_state or slot.split('-')[0] not in dial_domains:
                    turn_dialog_state[slot] = "none"
                else:
                    turn_dialog_state[slot] = normalize_label(slot, turn_dialog_state[slot])

                if turn_dialog_state[slot]=="dontcare":
                    turn_dialog_state[slot] = "do not care"

                ontology_modified[slot].append(turn_dialog_state[slot])
 
            dial_dict["dialogue"][ti]["belief_state"] = []

            # turn label
            turn_label = []
            for slot in slot_meta:
                if turn_dialog_state[slot] != prev_turn_state[slot]:
                    turn_label.append([slot, turn_dialog_state[slot]])
            dial_dict["dialogue"][ti]["turn_label"] = turn_label
                        
            prev_turn_state = turn_dialog_state
            
        dials_v2.append(dial_dict)
        
    with open(os.path.join(data_dir, file_id.split(".")[0]+"_v2.json"), 'w') as outfile:
        json.dump(dials_v2, outfile, indent=4)
                
# ontology extracted from dataset
for slot in slot_meta:
    ontology_modified[slot] = sorted(list(set(ontology_modified[slot])))
    
with open(os.path.join(data_dir, 'ontology-modified_v2.json'), 'w') as outfile:
     json.dump(ontology_modified, outfile, indent=4)