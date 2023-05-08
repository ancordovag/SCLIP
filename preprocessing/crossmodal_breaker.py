import json
import yaml
import os

def break_coco_files(data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    caption_file = os.path.join(data_dir, 'captions.jsonl')
    
    # Open the files
    train_file = open(os.path.join(out_dir, 'big_train_file.txt'), mode='w', encoding='utf-8')
    valid_file = open(os.path.join(out_dir, 'big_valid_file.txt'), mode='w', encoding='utf-8')
    
    N = 3600
    train_N = int(0.8*N)
    
    with open(caption_file,"r") as file:
        lines = file.readlines()
        print("Lines {}".format(len(lines)))

    count = 0
    train_count = 0
    for line in lines:
        jason = json.loads(line)
        caption = jason['en']["caption"][0]
        count += 1
        if caption == '':
            continue
        else:
            caption = caption.replace('\n','')        
        if count <= train_N:
            train_count += 1
            train_file.write(caption + '\n')
        else:
            valid_file.write(caption + '\n')

    train_file.close()
    valid_file.close()

    
if __name__ == "__main__":
    with open("preprocessing/config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    params = cfg["crossmodal"]
    data_dir = params['caption_dir']
    out_dir = params['data_dir']
    break_coco_files(data_dir, out_dir)
    