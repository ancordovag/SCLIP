import yaml
import os

if __name__ == '__main__':

    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    params = cfg["spanish_preprocessing"]
    data_dir = params["data_dir"]
    spanish_filename = os.path.join(data_dir, "Europarl.bg-es.es")

    out_dir = params["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Open the files
    train_file = open(os.path.join(out_dir, 'train_sentences.txt'), mode='w', encoding='utf-8')
    test_file = open(os.path.join(out_dir, 'test_sentences.txt'), mode='w', encoding='utf-8')
    valid_file = open(os.path.join(out_dir, 'valid_sentences.txt'), mode='w', encoding='utf-8')

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