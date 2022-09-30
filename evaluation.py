import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
import re
import time
from datetime import datetime
import yaml
import json
import os
from os.path import exists
import pandas as pd
from sentence_transformers import SentenceTransformer
from networks import SCLIPNN, SCLIPNN3
from utils import EmbeddingsDataset
import logging
from preprocessing.test_mixer import mix_test_files

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Models...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model.eval()

def get_test_file_path(directory):
    test_txt = 'test_sentences.txt'
    test_path  = os.path.join(directory, test_txt)
    return test_path
    
def get_sentences_from_file(filename):
    sentences = []
    with open(filename, mode='rt', encoding='utf-8') as file_object:
        for line in file_object:
            sentences.append(line)    
    return sentences

def regexification(sentences):
    regex = [r"[^A-Za-z0-9]+|[a-zA-Z][0-9]", r"(?<!\d)[0]\d*(?!\d)", r"\s+", r"[0-9]+"]
    for r in regex:
        sentences = list(map(lambda sentence: re.sub(r, " ", sentence), sentences))
    return sentences

def get_clip_embeddings(sentences, batch_size=16):
    tokenized_text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        clip_embeddings_list = []
        for i in range(0,tokenized_text.size()[0],batch_size):
            tok_batch = tokenized_text[i:i+batch_size]
            clip_embeddings_batch = clip_model.encode_text(tok_batch).to(device)
            for unity in clip_embeddings_batch:
                clip_embeddings_list.append(unity)
    final_emb = torch.stack(clip_embeddings_list)
    return final_emb

def get_sbert_embeddings(sentences):
    with torch.no_grad():  
        sbert_embeddings = torch.from_numpy(sbert_model.encode(sentences))
    return sbert_embeddings

def get_test_embeddings(path):
    if path == '':
        path = get_test_file_path('europarl')        
    test_sentences = regexification(get_sentences_from_file(path))
    print("CLIP encoding...")
    test_clip_embeddings = get_clip_embeddings(test_sentences)
    print("SBERT encoding...")
    test_embeddings = get_sbert_embeddings(test_sentences)
    return test_clip_embeddings, test_embeddings

def cosin_calculator(targets, predictions):    
    cosines = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for tar, pred in zip(targets, predictions):        
        cosine = cos(tar, pred)
        cosines.append(cosine.item())
    return np.array(cosines)

def evaluate(models, input_size, test_dataset,name_testset):
    batch_size = 4
    cosines = []
    euclideans = []
    with torch.no_grad():           
        for model in models:
            print("Evaluating model " + model)
            path = os.path.join('models',model)
            sp_name = model.split('_') 
            if 'NN3' in model:
                loaded_model = SCLIPNN3(input_size,int(sp_name[2])).to(device)
            else:
                loaded_model = SCLIPNN(input_size,int(sp_name[2])).to(device)
            loaded_model.load_state_dict(torch.load(path))            
            sum_cos = 0
            count = 0
            predictions =[]
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            for inputs, labels in test_loader:     
                tclip = labels.to(device)
                tsbert = inputs.to(device)
                prediction = loaded_model(tsbert)
                predictions.append(prediction)
                sum_cos += np.sum(cosin_calculator(tclip, prediction))
                count += 1
            n = len(test_dataset.X)
            this_cos = round(sum_cos/n,5)
            cosines.append(this_cos)
            stacked_predictions = torch.stack(predictions).view(-1,512)
            euclidean_dataset = EmbeddingsDataset(test_dataset.Y, stacked_predictions)
            euclidean_dataloader = DataLoader(euclidean_dataset, batch_size = batch_size)
            euc_sum = 0.0
            for tdata, spred in euclidean_dataloader:
                c_dist= torch.sum(torch.cdist(tdata.to(float), spred.to(float)))
                euc_sum += c_dist.item()
            #euclidean = torch.cdist(test_dataset.Y.to(float), stacked_predictions.to(float))
            avg_euclidean = euc_sum/n #torch.mean(euclidean)
            this_euc = round(avg_euclidean,5)
            euclideans.append(this_euc)   
            date_time = datetime.now()
            str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")            
            data_json = {"Timestamp":str_date_time, "name_model":sp_name[1]+'_'+sp_name[2], "hidden_layers":int(sp_name[2]), 
                         "epochs":int(sp_name[3].replace('e','')), "train_size":int(sp_name[4].replace('s','').replace('.pt','')), 
                         "train_dataset":sp_name[0], "test_dataset": name_testset, "model_saved_as": model, 
                         "cosine":this_cos, "euclidean":this_euc}  
            name_to_save = model.replace('.pt','.json')
            with open(os.path.join('evaluated',name_to_save), 'w', encoding='utf-8') as f:
                json.dump(data_json, f, ensure_ascii=False, indent=4)
    return cosines, euclideans                                   

def run_evaluation(directory='', n_epochs=0, testset='mixed'):
    test_file = ''
    if testset == 'mixed':
        test_file = 'test_mix.txt'
        if not exists(test_file):
            print("Creating file " + test_file)
            with open(os.path.join("preprocessing","config.yml"), "r") as ymlfile:
                cfg = yaml.safe_load(ymlfile)
                params = cfg["test"]
                size = params["size"]
                test_mixer.mix_test_files(params["size"], params["europarl_dir"], params["coco_dir"], params["out_dir"])
        else:
            with open('test_mix.txt', 'r') as file:
                lines = file.readlines()
                size = len(lines)
    models_to_evaluate = []
    all_models = os.listdir('models')
    for am in all_models:
        if directory != '' and directory not in am:
            continue
        if n_epochs > 0 and (e+str(n_epochs)) not in am:
            continue
        models_to_evaluate.append(am)
    print(f'Evaluating with test dataset {testset}...')    
    test_clip_emb, test_emb = get_test_embeddings(test_file)
    input_size = test_emb[0].size()[0]
    test_dataset = EmbeddingsDataset(test_emb, test_clip_emb)
    start_time = time.time()
    cosines, euclideans = evaluate(models_to_evaluate,input_size,test_dataset,name_testset=testset)
    end_time = time.gmtime(time.time() - start_time)
    evaluation_time = time.strftime("%H:%M:%S", end_time)  
    print("Evaluation Time: {}. Size: {}".format(evaluation_time),size)
    data = {"Cosin":cosines, "Euclidean":euclideans}
    indices = []
    for km in models_to_evaluate:
        indices.append(km)
    results = pd.DataFrame(data, index=indices)
    print(results)

if __name__ == "__main__":
    run_evaluation(directory='europarl')