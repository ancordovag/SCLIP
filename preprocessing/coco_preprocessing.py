import json
import yaml
import os


def write_coco_files(max_train, max_valid, max_test, data_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_train = os.path.join(data_dir, 'captions_train2017.json')
    file_val = os.path.join(data_dir, 'captions_val2017.json')
    
    f_train = open(file_train)  # 591753 lines
    train_data = json.load(f_train)
    train_file = open(os.path.join(out_dir, 'train_sentences.txt'), mode='w', encoding='utf-8')
    test_file = open(os.path.join(out_dir, 'test_sentences.txt'), mode='w', encoding='utf-8')
    count = 0
    train_count = 0
    for element in train_data["annotations"]:
        caption = element["caption"]
        if caption == '':
            continue
        else:
            caption = caption.replace('\n','')
        count += 1
        if count <= max_train:
            train_count += 1
            train_file.write(caption + '\n')
        elif count < max_train + max_test:
            test_file.write(caption + '\n')
        else:
            break
            
    # closing files
    f_train.close()
    train_file.close()
    test_file.close()
    
    f_val = open(file_val)
    val_data = json.load(f_val)  # 25014 lines
    valid_file = open(os.path.join(out_dir, 'valid_sentences.txt'), mode='w', encoding='utf-8')
    # Creating Validation file
    count = 0
    for element in val_data["annotations"]:
        caption = element["caption"]
        if caption == '':
            continue
        else:
            caption = caption.replace('\n','')
        count += 1
        valid_file.write(caption + '\n')
        if count >= max_valid:
            break
            
    # closing files
    f_val.close()
    valid_file.close()
    

if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    params = cfg["coco"]
    train_lines = params["max_train_lines"]    
    valid_lines = params["max_valid_lines"]
    test_lines = params["max_test_lines"]
    write_coco_files(train_lines, valid_lines, test_lines, params["caption_dir"], params["out_dir"])
