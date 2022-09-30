import yaml
import os

def write_files(max_train_lines, max_valid_lines, data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Open the files
    train_file = open(os.path.join(out_dir, 'train_sentences.txt'), mode='w', encoding='utf-8')
    valid_file = open(os.path.join(out_dir, 'valid_sentences.txt'), mode='w', encoding='utf-8')

    train_english_filename = os.path.join(data_dir, 'coco_train.txt')
    valid_english_filename = os.path.join(data_dir, 'coco_valid.txt')

    lines_counter = 0
    with open(train_english_filename, mode='rt', encoding='utf-8') as english_file:
        lines = english_file.readlines()
        for english_line in lines:
            lines_counter += 1
            if lines_counter > max_train_lines:
                break
            train_file.write(english_line.rstrip('\n') + '\n')
    
    lines_counter = 0
    with open(valid_english_filename, mode='rt', encoding='utf-8') as english_file:
        lines = english_file.readlines()
        for english_line in lines:
            lines_counter += 1
            if lines_counter > max_valid_lines:
                break
            valid_file.write(english_line.rstrip('\n') + '\n')
        
    train_file.close()
    valid_file.close()
    
if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    params = cfg["coco"]
    train_lines = params["max_train_lines"]    
    valid_lines = params["max_valid_lines"]
    write_files(train_lines, valid_lines, params["out_dir"], params["out_dir"])

    