import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import clip
import re
import time
import yaml
from matplotlib import pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from networks import SCLIPNN, SCLIP_ACTIV

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Models")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#Our sentences we like to encode
dirname = 'europarl/'
train_path = 'train_sentences.txt'
test_path = 'test_sentences.txt'
valid_path = 'valid_sentences.txt'
train_filename = dirname + '/' + train_path
test_filename = dirname + '/' + test_path
valid_filename = dirname + '/' + valid_path

train_sentences = []
with open(train_filename, mode='rt', encoding='utf-8') as file_object:
    for line in file_object:
        train_sentences.append(line)
N = len(train_sentences)
print("Number of sentences to train: {}".format(N))

regex = [r"[^A-Za-z0-9]+|[a-zA-Z][0-9]", r"(?<!\d)[0]\d*(?!\d)", r"\s+", r"[0-9]+"]
for r in regex:
    train_sentences = list(map(lambda sentence: re.sub(r, " ", sentence), train_sentences))
    
text = clip.tokenize(train_sentences).to(device)

print("CLIP encoding...")
with torch.no_grad():
    clip_embeddings = clip_model.encode_text(text)

print("SBERT encoding...")
with torch.no_grad():  
    sbert_embeddings = torch.from_numpy(sbert_model.encode(train_sentences))
    
#Print the embeddings
print("-"*10)
for sentence, clip_embedding, sbert_embedding in zip(train_sentences[:1], clip_embeddings[:1], sbert_embeddings[:1]):
    print("Sentence:", sentence)
    print("Clip Embedding: ", clip_embedding.size())
    print("Sbert Embedding: ", sbert_embedding.size())
    print("-"*10)
    
print("Creating Models to train")
model_NN_1 = SCLIPNN(200).to(device)
model_NN_2 = SCLIPNN(200).to(device)
model_NORM_1 = SCLIP_ACTIV(200, "leak").to(device)
model_NORM_2 = SCLIP_ACTIV(200, "leak").to(device)

models = {'NN1':model_NN_1, 'NN2':model_NN_2,
         'NORM1':model_NORM_1, 'NORM2':model_NORM_2}

with open("config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
params = cfg["training"]

def train(model, sbert_emb, clip_emb, epochs=params["epochs"], print_every=params["print_every"]):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
    losses = []
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
        for i, data in enumerate(zip(sbert_emb, clip_emb)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.to(float), labels.to(float))       
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        losses.append(epoch_loss)

        if epoch % print_every == 0:
            print("Epoch {}. Loss: {}".format(epoch, epoch_loss))
    
    print("Final Loss: {}".format(losses[-1]))
    
    return losses

losses = []
final_loss = []
training_time = []
for name, model in models.items():
    start_time = time.time()
    print('Training model {}'.format(name))
    loss = train(model, sbert_embeddings, clip_embeddings)
    losses.append(loss)    
    final_loss.append(round(loss[-1],3))
    end_time = time.time() - start_time
    end_time = time.gmtime(end_time)
    elapsed_time = time.strftime("%H:%M:%S", end_time)
    training_time.append(elapsed_time)
    print('Finished Training from model {}. Elapsed time: {}.'.format(name,elapsed_time))
    print("-"*50)
actual_time = time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(time.time()))
print("End of Training Process on {}".format(actual_time))

test_sentences = []
with open(test_filename, mode='rt', encoding='utf-8') as file_object:
    for line in file_object:
        test_sentences.append(line)
N = len(test_sentences)
print("Number of sentences to test: {}".format(N))

for r in regex:
    test_sentences = list(map(lambda sentence: re.sub(r, " ", sentence), test_sentences))
    
text = clip.tokenize(test_sentences).to(device)
with torch.no_grad():
    test_clip_embeddings = clip_model.encode_text(text)
    
with torch.no_grad():
    test_sbert_embeddings = torch.from_numpy(sbert_model.encode(test_sentences))
    
def cosin_calculator(targets, predictions):    
    cosines = []
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for tar, pred in zip(targets, predictions):        
        cosine = cos(tar, pred)
        cosines.append(cosine.item())
    return np.array(cosines)

print("Evaluating...")
cosines = []
euclideans = []
with torch.no_grad():
    for name, model in models.items():
        sum_cos = 0
        count = 0
        predictions =[]
        if len(test_clip_embeddings) == 0:
            break
        for tclip, tsbert in zip(test_clip_embeddings, test_sbert_embeddings):        
            tclip = tclip.to(device)
            tsbert = tsbert.to(device)
            prediction = model(tsbert)
            predictions.append(prediction)
            sum_cos += np.mean(cosin_calculator(tclip, prediction))
            count += 1
        cosines.append(round(sum_cos/count,3))
        stacked_predictions = torch.stack(predictions)
        euclidean = torch.cdist(test_clip_embeddings.to(float), stacked_predictions.to(float))
        avg_euclidean = torch.mean(euclidean)
        euclideans.append(round(avg_euclidean.item(),3))    
        
data = {"Cosin":cosines, "Euclidean":euclideans, 
        "TrainTime":training_time, "Loss":final_loss}
results = pd.DataFrame(data, index=models.keys())

print(results)