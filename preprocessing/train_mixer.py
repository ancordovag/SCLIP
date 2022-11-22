import json
import os
import random
import yaml

def mix_files(europarl_dir, coco_dir, out_dir):
    
    europarl_train_path = os.path.join(europarl_dir,'train_sentences.txt')
    europarl_valid_path = os.path.join(europarl_dir,'valid_sentences.txt')
    coco_train_path = os.path.join(coco_dir,'train_sentences.txt')
    coco_valid_path = os.path.join(coco_dir,'valid_sentences.txt')
    final_train_path = os.path.join(out_dir,'train_sentences.txt')
    final_valid_path = os.path.join(out_dir,'valid_sentences.txt')
    
    all_train_lines = []
    all_valid_lines = []
    
    with open(europarl_train_path, mode='r', encoding='utf-8') as file:
        euro_lines = file.readlines()
        for line in euro_lines:
            line = line.replace('\n','')
            all_train_lines.append(line+'\n')
    
    with open(coco_train_path, mode='r', encoding='utf-8') as file:
        coco_lines = file.readlines()
        for line in coco_lines:
            line = line.replace('\n','')
            all_train_lines.append(line+'\n')
    
    with open(europarl_valid_path, mode='r', encoding='utf-8') as file:
        euro_lines = file.readlines()
        for line in euro_lines:
            line = line.replace('\n','')
            all_valid_lines.append(line+'\n')
    
    with open(coco_valid_path, mode='r', encoding='utf-8') as file:
        coco_lines = file.readlines()
        for line in coco_lines:
            line = line.replace('\n','')
            all_valid_lines.append(line+'\n')
    
    count = 0
    with open(final_train_path, mode='w', encoding='utf-8') as file:        
        for atl in all_train_lines:
            count += 1
            file.write(atl)
    print(f"Mixed Training Lines: {count}")
            
    count = 0
    with open(final_valid_path,mode='w',encoding='utf-8') as file:        
        for atl in all_valid_lines:
            count += 1
            file.write(atl)
    print(f"Mixed Validation Lines: {count}")
    
if __name__ == "__main__":
    with open(os.path.join("preprocessing","config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    params = cfg["mix"]
    mix_files(params["europarl_dir"], params["coco_dir"], params["out_dir"])