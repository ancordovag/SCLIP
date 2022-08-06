import json
import os

def write_test_file(max_test):
    europarl_path = os.path.join('europarl','Europarl.en-es.en')
    coco_path = os.path.join('coco','captions_train2017.json')
    
    coco_file = open(coco_path)
    coco_data = json.load(coco_file)
    
    
    mixed_sentences = []
    offset = 200000
        
    # First add coco sentences
    count = 0
    for element in coco_data["annotations"]:        
        caption = element["caption"]
        if caption == '':
            continue
        else:
            caption = caption.replace('\n','')
        count += 1
        if count < offset:
            continue
        if count >= offset + int(max_test/2):
            break
        mixed_sentences.append(caption+'\n')
            
    # Then europarl sentences
    lines_counter = 0
    max_sentence_length = 30
    with open(europarl_path, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines[offset:offset+max_test]:
            number_words = len(line.split())
            if number_words > max_sentence_length:
                continue
            lines_counter += 1
            if lines_counter <= int(max_test/2):
                line = line.replace('\n','')
                mixed_sentences.append(line+'\n')
            else:
                break
    
    with open('mixed_test_sentences_'+str(max_test)+'.txt',mode='w',encoding='utf-8') as file:
        file.writelines(mixed_sentences)
        
if __name__ == "__main__":
    write_test_file(10000)