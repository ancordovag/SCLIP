import yaml

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

params = cfg["spanishpreprocessing"]

dirname = 'europarl_es/'    
spanish_dir = 'Europarl.bg-es.es'
spanish_filename = dirname + '/' + spanish_dir

# Open the files
train_file = open(dirname + 'train_sentences.txt',mode='w',encoding='utf-8')
test_file = open(dirname + 'test_sentences.txt',mode='w',encoding='utf-8')
valid_file = open(dirname + 'valid_sentences.txt',mode='w',encoding='utf-8')
max_training_lines = params["max_train_lines"]
max_testing_lines = params["max_test_lines"]
max_validation_lines = params["max_valid_lines"]
max_sentence_length = params["max_sentence_length"]

lines_counter = 0
with open(spanish_filename, mode='rt', encoding='utf-8') as file:
    for line in file:
        line = file.readline().rstrip('\n')
        number_words = len(line.split())
        if number_words < max_sentence_length:
            lines_counter += 1
            if lines_counter <= max_training_lines:
                train_file.write(line + '\n')
                continue
            if lines_counter <= max_training_lines + max_testing_lines:
                test_file.write(line + '\n')
                continue
            valid_file.write(line + '\n')
            if lines_counter >= max_training_lines + max_testing_lines + max_validation_lines:
                break     

train_file.close()
test_file.close()
valid_file.close()