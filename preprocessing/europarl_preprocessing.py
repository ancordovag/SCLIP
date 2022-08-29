import yaml
import os


def write_new_files(max_train_lines, max_valid_lines, max_test_lines, data_dir, out_dir, max_sentence_length=30):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Open the files
    train_file = open(os.path.join(out_dir, 'train_sentences.txt'), mode='w', encoding='utf-8')
    valid_file = open(os.path.join(out_dir, 'valid_sentences.txt'), mode='w', encoding='utf-8')
    test_file = open(os.path.join(out_dir, 'test_sentences.txt'), mode='w', encoding='utf-8')

    english_filename = os.path.join(data_dir, 'Europarl.en-es.en')

    lines_counter = 0
    with open(english_filename, mode='rt', encoding='utf-8') as english_file:
        for english_line in english_file:
            english_line = english_file.readline().rstrip('\n')
            number_words = len(english_line.split())
            if number_words < max_sentence_length:
                lines_counter += 1
                if lines_counter <= max_train_lines:
                    train_file.write(english_line + '\n')
                    continue
                if lines_counter <= max_train_lines + max_valid_lines:
                    valid_file.write(english_line + '\n')
                    continue

                test_file.write(english_line + '\n')
                if lines_counter >= max_train_lines + max_test_lines + max_valid_lines:
                    break     

    train_file.close()
    test_file.close()
    valid_file.close()


if __name__ == "__main__":
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    params = cfg["preprocessing"]
    dirname = 'europarl/'    
    train_lines = params["max_train_lines"]
    valid_lines = params["max_valid_lines"]
    test_lines = params["max_test_lines"]
    sentence_length = params["max_sentence_length"]
    write_new_files(train_lines, valid_lines, test_lines, params["data_dir"], params["out_dir"], sentence_length)