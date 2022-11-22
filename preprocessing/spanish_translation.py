import time
from googletrans import Translator
import yaml
from pathlib import Path
import os

def write_translated_file(lines, path, code):
    print(f'Starting translation to language {code}')
    print(f'Final Path: {path}')
    print(f'Length of lines: {len(lines)}')
    translator = Translator()
    with open(path,'w') as file:
        i = 0
        for line in lines:
            i += 1
            translation = translator.translate(line, dest=code)
            new_caption = translation.text
            file.write(new_caption.rstrip('\n') + '\n')
            if i % 1000 == 0:
                print("Counter {}: Caption: {}  --> Translation: {}".format(i, line.rstrip('\n'), new_caption.rstrip('\n')))
            time.sleep(0.04)
    print(f'{i} lines written in file {path}')

if __name__ == '__main__':

    with open(os.path.join("preprocessing","config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    directory = cfg["coco"]["out_dir"]
    esco_dir = cfg["esco"]["out_dir"]

    # source data
    #with open(os.path.join(directory,'big_train_file.txt'),'r') as file:
    #    train_lines = file.readlines()
    #print(f'Length of training lines: {len(train_lines)}')
    
    with open(os.path.join(directory,'big_valid_file.txt'),'r') as file:
        valid_lines = file.readlines()
    
    with open(os.path.join(directory,'big_test_file.txt'),'r') as file:
        test_lines = file.readlines()
        
    code = "es"
    
    #train_path = os.path.join(esco_dir, 'big_train_file.txt')
    valid_path = os.path.join(esco_dir, 'big_valid_file.txt')
    test_path = os.path.join(esco_dir, 'big_test_file.txt')
        
    write_translated_file(valid_lines, valid_path, code)
    print('-'*200)
    write_translated_file(test_lines, test_path, code)
    print('-'*200)
    #write_translated_file(train_lines, train_path, code)
    #print('-'*200)
   
    print("Done")
