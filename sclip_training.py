import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import clip
import re
import time
import yaml
import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
from networks import SCLIPNN, SCLIPNN3
from utils import EmbeddingsDataset, get_models_to_train
import logging
from datetime import datetime
from experiment import get_MRR, get_image_and_captions_clip_features

logger = logging.getLogger(__name__)
fhandler = logging.FileHandler(filename='trained_models.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading Models...")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model.eval()
print("Models Loaded: CLIP, SBERT")

# TODO: we should have a different file for the experiments and load it once from the main function
# TODO: duplicate fragment from experiment.py
with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
pairs_directory = cfg["coco"]["out_dir"]
image_directory = cfg["coco"]["image_dir"]
languages = cfg["languages"]


def get_files_paths(directory):
    train_txt = 'train_sentences.txt'    
    valid_txt = 'valid_sentences.txt'
    train_path = os.path.join(directory, train_txt)
    valid_path = os.path.join(directory, valid_txt)
    return train_path, valid_path


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


def get_clip_embeddings(sentences, batch_size=32):
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


def show_embeddings_return_size(sentences, clip_embeddings, sbert_embeddings):
    for sentence, clip_embedding, sbert_embedding in zip(sentences[:1], clip_embeddings[:1], sbert_embeddings[:1]):
        print("Sentence:", sentence)
        input_size = sbert_embedding.size()[0]    
        print("Sbert Embedding: ", input_size)
        print("Clip Embedding: ", clip_embedding.size()[0])
        print("-"*10)
    return input_size


def show_plot(models, model_train_losses, model_valid_losses, plot_name):
    n = len(models)
    rows = math.ceil(n/4)
    columns = math.ceil(n/rows)
    fig, axs = plt.subplots(rows, columns, figsize=(18, 10))
    positions = []
    if rows == 1:
        for c in range(columns):
            positions.append(c)
    else:
        for r in range(rows):
            for c in range(columns):
                positions.append((r, c))
        
    if n == 1:
        for i, (name, model) in enumerate(models.items()):
            axs.plot(model_train_losses[i], label='train ' + name)
            axs.plot(model_valid_losses[i], label='valid ' + name, marker='*')
            axs.set_title('Losses of '+ name)
            axs.grid()
            axs.legend()
            axs.set(xlabel='Epochs', ylabel='Loss')
    else:
        for i, (name, model) in enumerate(models.items()):
            axs[positions[i]].plot(model_train_losses[i], label='train ' + name)
            axs[positions[i]].plot(model_valid_losses[i], label='valid ' + name, marker='*')
            axs[positions[i]].set_title('Losses of '+ name)
            axs[positions[i]].grid()
            axs[positions[i]].legend()
        for ax in axs.flat:
            ax.set(xlabel='Epochs', ylabel='Loss')

    # fig.legend()
    # fig.title('Losses per Epoch')

    plt.savefig(os.path.join('imgs', '{}.png'.format(plot_name)))
    plt.show()


def train(model, train_dataset, valid_dataset, b_size=32, epochs=200):
    images_features, clip_features, captions = get_image_and_captions_clip_features(languages, image_directory,clip_model, preprocess)
    
    writer = SummaryWriter()  # TODO argument log_dir= should be a folder out of the project dir
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=b_size, shuffle=True)
        train_counter = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.to(float), labels.to(float))       
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_counter += 1
        train_avg_loss = train_loss/train_counter
        train_losses.append(train_avg_loss)
        writer.add_scalar("Loss/Train", train_avg_loss, epoch)
        writer.flush()
        
        valid_loss = 0.0
        model.eval()
        valid_counter = 0
        sbert_per, clip_per, sbert_MRR, clip_MRR, sbert_er, clip_er = get_MRR(model,pairs_directory,languages,sbert_model,captions, images_features,clip_features)
        for i, (lang, code) in enumerate(languages.items()):
            writer.add_scalar(lang+"/Performance/SBERT", sbert_per[i], epoch)
            # writer.add_scalar(lang+"/Performance/CLIP", clip_per[i], epoch)
            writer.add_scalar(lang+"/MRR/SBERT", sbert_MRR[i], epoch)
            # writer.add_scalar(lang+"/MRR/CLIP", clip_MRR[i], epoch)
            writer.add_scalar(lang+"/Error/SBERT", sbert_er[i], epoch)
            # writer.add_scalar(lang+"/Error/CLIP", clip_er[i], epoch)
            
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            target = model(inputs)
            loss = criterion(target.to(float), labels.to(float))
            valid_loss += loss.item()
            valid_counter += 1
        valid_avg_loss = valid_loss/valid_counter    
        valid_losses.append(valid_avg_loss)
        print("Epoch {}. Train Loss: {}. Valid Loss: {}".format(epoch, train_loss/train_counter, valid_loss/valid_counter))
        
        writer.add_scalar("Loss/Valid", valid_avg_loss, epoch)
        writer.flush()
        
    writer.close()        
        
    return train_losses, valid_losses

def supra_training(models,train_sbert_emb,train_clip_emb,valid_sbert_emb, valid_clip_emb, size, trainset,n_epochs):
    model_train_losses = []
    model_valid_losses = []
    final_loss = []
    training_time = []
    for name, model in models.items():
        start_time = time.time()
        train_dataset = EmbeddingsDataset(train_sbert_emb, train_clip_emb)
        valid_dataset = EmbeddingsDataset(valid_sbert_emb, valid_clip_emb)
        train_loss, valid_loss = train(model, train_dataset, valid_dataset, epochs=n_epochs)
        end_time = time.gmtime(time.time() - start_time)
        elapsed_time = time.strftime("%H:%M:%S", end_time)
        training_time.append(elapsed_time)
        model_train_losses.append(train_loss)                   
        model_valid_losses.append(valid_loss)
        final_loss.append(round(valid_loss[-1],5))
        date_time = datetime.now()
        str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
        name_to_save = trainset + '_' + name +'_e'+str(n_epochs)+'_s'+str(size)+ '.pt'
        #data_json = {"Timestamp":str_date_time, "name_model":name, "hidden_layers":int(name.split("_")[1]), 
        #             "epochs":n_epochs, "train_size":size, "train_dataset":trainset, "model_saved_as": name_to_save + '.pt',
        #             "train_loss":round(train_loss[-1],3), "valid_loss":round(valid_loss[-1],3), "elapsed_time": elapsed_time}        
        #with open(os.path.join('jsons',name_to_save+'.json'), 'w', encoding='utf-8') as f:
        #    json.dump(data_json, f, ensure_ascii=False, indent=4)
        logger.info(f'Trained model called {name_to_save} at {str_date_time}')
        if os.path.exists("models"):  # TODO: this should be out of the project folder
            os.makedirs("models")
        torch.save(model.state_dict(), os.path.join('models', name_to_save))
        print('Finished Training from model {}. Elapsed time: {}.'.format(name,elapsed_time))
        # print("-"*50)
    actual_time = time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(time.time()))
    print("End of Training Process on {}".format(actual_time))
    return model_train_losses, model_valid_losses, training_time, final_loss


def get_train_embeddings(directory):
    train_file, valid_file = get_files_paths(directory)
    train_sentences = regexification(get_sentences_from_file(train_file))
    valid_sentences = regexification(get_sentences_from_file(valid_file))
    print("CLIP encoding...")
    train_clip_embeddings = get_clip_embeddings(train_sentences)
    valid_clip_embeddings = get_clip_embeddings(valid_sentences)
    print("SBERT encoding...")
    train_sbert_embeddings = get_sbert_embeddings(train_sentences)
    valid_sbert_embeddings = get_sbert_embeddings(valid_sentences)
    return train_clip_embeddings, valid_clip_embeddings, train_sbert_embeddings, valid_sbert_embeddings


def run_pipeline(directory, n_epochs):  # Training Pipeline
    durations = {}
    finals = {}
    model_dict = {}
    train_clip_emb, valid_clip_emb, train_sbert_emb, valid_sbert_emb = get_train_embeddings(directory)
    train_size = train_sbert_emb.size()[0]
    input_size = train_sbert_emb.size()[1]
    # print(f'DEBUG: Train Size: {train_size}. Input Size: {input_size}')
    model_dict[directory] = get_models_to_train(input_size)
    # print(f'Training on {directory} dataset')
    # print("Train sbert_ emb : {}".format(train_sbert_emb.size()))
    # print("Train clip emb: {}".format(train_clip_emb.size()))
    train_losses, valid_losses, train_time, final_loss = supra_training(
        model_dict[directory],
        train_sbert_emb,
        train_clip_emb,
        valid_sbert_emb,
        valid_clip_emb,
        train_size,
        trainset=directory,
        n_epochs=n_epochs
    )
    durations[directory] = train_time
    finals[directory] = final_loss
    train_final_losses = [x[-1] for x in train_losses]
    train_results = pd.DataFrame({"TrainLoss": train_final_losses, "ValidLoss": final_loss}, index=model_dict[directory].keys())
    print(train_results)
    show_plot(model_dict[directory], train_losses, valid_losses, directory+'_'+str(n_epochs)+'_'+str(train_size))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SCLIP")
    parser.add_argument("--dataset", help="decide which dataset to use", default="europarl", type=str)
    parser.add_argument('--epochs', type=int, default=100, help='how many epoches for this training')
    args = parser.parse_args()

    run_pipeline(args.dataset, args.epochs)
