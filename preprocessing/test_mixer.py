import json
import os
import random
import yaml

def mix_test_files(size, europarl_dir, coco_dir, out_dir):
    europarl_path = os.path.join(europarl_dir,'test_sentences.txt')
    coco_path = os.path.join(coco_dir,'test_sentences.txt')
    final_path = os.path.join(out_dir,'test_mix.txt')
    
    all_lines = []
    mixed_sentences = []
    
    with open(europarl_path, mode='r', encoding='utf-8') as file:
        euro_lines = file.readlines()
        for line in euro_lines:
            line = line.replace('\n','')
            all_lines.append(line+'\n')
            all_lines.append(line)
                        
    with open(coco_path, mode='r', encoding='utf-8') as file:
        coco_lines = file.readlines()
        for line in coco_lines:
            line = line.replace('\n','')
            all_lines.append(line+'\n')
            
    N_all = len(all_lines)
    if  N_all < size:
        print(f'Number of test sentences {N_all} is smaller than required size {size}. Using all test sentences.')
        mixed_sentences = all_lines
    else:
        N_mixed = 0
        while(N_mixed < size):
            selected = random.choice(all_lines)
            mixed_sentences.append(selected)
            all_lines.remove(selected)
            N_mixed += 1            
    
    with open(final_path,mode='w',encoding='utf-8') as file:
        for ms in mixed_sentences:
            file.write(ms)
        
if __name__ == "__main__":
    with open(os.path.join("preprocessing","config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    params = cfg["test"]
    size = params["size"]
    mix_test_files(params["size"], params["europarl_dir"], params["coco_dir"], params["out_dir"])