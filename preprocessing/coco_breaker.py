import json
import yaml
import os

def break_coco_files(data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_train = os.path.join(data_dir, 'captions_train2017.json')
    file_val = os.path.join(data_dir, 'captions_val2017.json')
    
    # Open the files
    train_file = open(os.path.join(out_dir, 'big_train_file.txt'), mode='w', encoding='utf-8')
    valid_file = open(os.path.join(out_dir, 'big_valid_file.txt'), mode='w', encoding='utf-8')
    test_file = open(os.path.join(out_dir, 'big_test_file.txt'), mode='w', encoding='utf-8')
    
    # Separate the train annotations in train and validation
    f_train = open(file_train)
    train_data = json.load(f_train)
    count = 0
    train_count = 0
    N = len(train_data["annotations"])
    train_N = int(0.8*N)
    for element in train_data["annotations"]:
        caption = element["caption"]
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
            
    # closing files
    f_train.close()
    train_file.close()
    valid_file.close()
    
    # Transform the validation file in a test file
    f_val = open(file_val)
    test_data = json.load(f_val)  # 25014 lines
    # Creating Validation file
    count = 0
    for element in test_data["annotations"]:
        caption = element["caption"]
        count += 1
        if caption == '':
            continue
        else:
            caption = caption.replace('\n','')        
        test_file.write(caption + '\n')
            
    # closing files
    f_val.close()
    test_file.close()
    
if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    params = cfg["coco"]
    data_dir = params['caption_dir']
    out_dir = params['data_dir']
    break_coco_files(data_dir, out_dir)
    