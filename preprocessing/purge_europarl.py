import clip
import os, yaml
import re
import torch
import time

def get_clip_embeddings(sentences):
    tokenized_text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        clip_embeddings = clip_model.encode_text(tokenized_text)
        clip_embeddings.to('cpu')
    return clip_embeddings

def length_sentence(sentence):
    splitted = sentence.split()
    return len(splitted)

def truncate_sentence(sentence, length=60):
    splitted = sentence.split()
    N = len(splitted)
    if  N < length:
        length = N
    result = ""
    for i in range(length):
        result += splitted[i] + " "
    result = result[:-1]
    rest = ""
    for i in range(length,N):
        rest += splitted[i] + " "
    rest = rest[:-1]
    return result, rest

def sentence_regexification(sentence):
    regex = [r"[^A-Za-z0-9]+|[a-zA-Z][0-9]", r"(?<!\d)[0]\d*(?!\d)", r"\s+", r"[0-9]+"]
    for r in regex:
        sentence = re.sub(r, " ", sentence)
    return sentence

def get_number_of_lines(path_to_file):
    with open(path_to_file,'r') as file:
        lines = file.readlines()
    return len(lines)

if __name__ == "__main__":
    start_time = time.time()
    with open(os.path.join("preprocessing","config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    params = cfg["preprocessing"]
    data_dir = params["data_dir"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    original_europarl = os.path.join(data_dir, 'Europarl.en-es.en')
    new_europarl = os.path.join(data_dir, 'Europarl-en.txt')
    long_europarl = os.path.join(data_dir, 'Europarl_long-en.txt')
    removed_europarl = os.path.join(data_dir, 'Europarl_removed-en.txt')

    print('Loading CLIP')
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    new_lines = []
    print(f'Reading Europarl from : {original_europarl}')

    with open(original_europarl,'r') as file:
        lines = file.readlines()
        to_remove = []
        long_sentences = []
        long_lines = 0
        count = 0
        i = 0
        for line in lines:
            i += 1
            line = sentence_regexification(line).rstrip('\n')
            s_length = length_sentence(line)
            if i % 100000 == 0:
                middle_time = time.gmtime(time.time() - start_time)
                elaps_time = time.strftime("%H:%M:%S", middle_time)
                print(f'Line {i} of length {s_length} after time {elaps_time}: {line}')            
            if s_length < 25:
                new_lines.append(line)
                continue
            truncator = 55
            if s_length > truncator:
                ## TODO: Put in a separate file to encode during training
                long_sentences.append(line)
                long_lines += 1
                continue            
            try:
                emb = get_clip_embeddings([line])
                new_lines.append(line)
            except Exception as e:
                middle_time = time.gmtime(time.time() - start_time)
                elaps_time = time.strftime("%H:%M:%S", middle_time)
                print(f'Line {i} of length {s_length} after time {elaps_time}: EXCEPTION')  
                print(f'Exception number {count}: {str(e)}')
                print('--'*40)
                to_remove.append(line)
                count +=1
    print(f'Number of iterations: {i}')

    print(f'Long lines: {long_lines}')
    with open(long_europarl,'w') as file:
        for ll in long_sentences:
            file.write(ll+'\n')
            
    print(f'Removed lines: {count}')
    with open(removed_europarl,'w') as file:
        for tr in to_remove:
            file.write(tr+'\n')
    
    print(f'Safe lines: {len(new_lines)}')
    with open(new_europarl, 'w') as file:
        for new_line in new_lines:
            file.write(new_line+'\n')

    print(f'Lines in purged Europarl : {get_number_of_lines(new_europarl)}')
    print(f'Long lines in file: {long_europarl}')
    print(f'Removed lines in file: {removed_europarl}')
    print(f'New safe corpus: {new_europarl}')

    end_time = time.gmtime(time.time() - start_time)
    elapsed_time = time.strftime("%H:%M:%S", end_time)

    print(f'Total time: {elapsed_time}')