import yaml
import os

def write_files(directory, train_lines, valid_lines, data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Open the files
    train_file = open(os.path.join(out_dir, 'train_sentences.txt'), mode='w', encoding='utf-8')
    valid_file = open(os.path.join(out_dir, 'valid_sentences.txt'), mode='w', encoding='utf-8')

    train_english_filename = os.path.join(data_dir, 'big_train_file.txt')
    valid_english_filename = os.path.join(data_dir, 'big_valid_file.txt')

    lines_counter = 0
    with open(train_english_filename, mode='rt', encoding='utf-8') as english_file:
        lines = english_file.readlines()
        for english_line in lines:
            lines_counter += 1
            if lines_counter > train_lines:
                break
            train_file.write(english_line.rstrip('\n') + '\n')
    
    lines_counter = 0
    with open(valid_english_filename, mode='rt', encoding='utf-8') as english_file:
        lines = english_file.readlines()
        for english_line in lines:
            lines_counter += 1
            if lines_counter > valid_lines:
                break
            valid_file.write(english_line.rstrip('\n') + '\n')
        
    train_file.close()
    valid_file.close()
    
if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    directory = 'europarl'
    params = cfg[directory]
    train_lines = params["max_train_lines"]    
    valid_lines = params["max_valid_lines"]
    write_files(directory,train_lines, valid_lines, params["data_dir"], params["out_dir"])