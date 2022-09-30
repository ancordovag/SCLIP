import os, yaml

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

params = cfg["europarl"]
data_dir = params["data_dir"]

english_filename = os.path.join(data_dir, 'Europarl-en.txt')

# Open the files
train_file = open(os.path.join(data_dir, 'big_train_file.txt'), mode='w', encoding='utf-8')
valid_file = open(os.path.join(data_dir, 'big_valid_file.txt'), mode='w', encoding='utf-8')
test_file = open(os.path.join(data_dir, 'big_test_file.txt'), mode='w', encoding='utf-8')

with open(english_filename, mode = 'r', encoding='utf-8') as english_file:
    lines = english_file.readlines()
    total_lines = len(lines)

max_train_lines = int(total_lines*0.9*0.8)
max_valid_lines = int(total_lines*0.9*0.2)
max_test_lines = int(total_lines*0.1)

print("Sizes: Train {}; Valid {}; Test {}".format(max_train_lines, max_valid_lines, max_test_lines))

lines_counter = 0
with open(english_filename, mode='rt', encoding='utf-8') as english_file:
    lines = english_file.readlines()
    for english_line in lines:
        lines_counter += 1
        if lines_counter <= max_train_lines:
            train_file.write(english_line.rstrip('\n') + '\n')
            continue
        if lines_counter <= max_train_lines + max_valid_lines:
            valid_file.write(english_line.rstrip('\n') + '\n')
            continue
        test_file.write(english_line.rstrip('\n') + '\n')

print("Lines written: {}".format(lines_counter))
        
train_file.close()
test_file.close()
valid_file.close()
