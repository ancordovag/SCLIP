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
import sys
import wandb
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
from utils import EmbeddingsDataset, get_models_to_train
import logging
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_NOTEBOOK_NAME"] = "sclip_training"
wandb.init(project="test-project", entity="sclip")
logger = logging.getLogger(__name__)
fhandler = logging.FileHandler(filename='trained_models.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

def get_sbert_and_clip_models():
    print("Loading Models...")
    sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("SBERT model loaded")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded")
    return sbert_model.eval(), clip_model.eval(), preprocess

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

def get_images_and_captions(directory, languages):
    images_of_language = {}
    captions_of_language = {}
    for lang, code in languages.items():        
        f_json = open(os.path.join(directory, "{}_pairs.json".format(code)), mode='r', encoding='utf-8')
        pairs_data = json.load(f_json)
        images = []
        captions = []
        for pair in pairs_data:
            images.append(pair["image_id"])
            captions.append(pair["caption"])
        images_of_language[lang] = images
        captions_of_language[lang] = captions
    return images_of_language, captions_of_language

def regexification(sentences):
    regex = [r"[^A-Za-z0-9]+|[a-zA-Z][0-9]", r"(?<!\d)[0]\d*(?!\d)", r"\s+", r"[0-9]+"]
    for r in regex:
        sentences = list(map(lambda sentence: re.sub(r, " ", sentence), sentences))
    return sentences

def truncate_sentence(sentence, length=55):
    new_sentences = []
    splitted = sentence.split()
    N = len(splitted)
    if  N < length:
        new_sentences.append(sentence)
        return new_sentences
    
    result = ""
    for i in range(length):
        result += splitted[i] + " "
    result = result[:-1]
    new_sentences.append(result)
                         
    rest = ""
    for i in range(length,N):
        rest += splitted[i] + " "
    rest = rest[:-1]
    other_sentences = truncate_sentence(rest)
    for os in other_sentences:
        new_sentences.append(os)
    return new_sentences

def get_clip_embeddings(sentences, clip_model, batch_size=32):
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

def get_clip_long_embeddings(sentences, batch_size=32):
    clip_embeddings_list = []
    for sentence in sentences:
        short_sentences = truncate_sentence(sentence)
        tokenized_text = clip.tokenize(short_sentences).to(device)
        with torch.no_grad():            
            clip_embeddings_sentences = clip_model.encode_text(tokenized_text).to(device)
            clip_embeddings = torch.mean(clip_embeddings_sentences,0)
            clip_embeddings_list.append(clip_embeddings)
    final_emb = torch.stack(clip_embeddings_list)
    return final_emb

def get_sbert_embeddings(sentences, sbert_model):
    with torch.no_grad():  
        sbert_embeddings = torch.from_numpy(sbert_model.encode(sentences))
    return sbert_embeddings


def get_train_embeddings(directory, clip_model, sbert_model):
    train_file, valid_file = get_files_paths(directory)
    train_sentences = regexification(get_sentences_from_file(train_file))
    valid_sentences = regexification(get_sentences_from_file(valid_file))
    print("CLIP encoding...")
    train_clip_embeddings = get_clip_embeddings(train_sentences, clip_model)
    valid_clip_embeddings = get_clip_embeddings(valid_sentences, clip_model)
    print("SBERT encoding...")
    train_sbert_embeddings = get_sbert_embeddings(train_sentences, sbert_model)
    valid_sbert_embeddings = get_sbert_embeddings(valid_sentences, sbert_model)
    return train_clip_embeddings, valid_clip_embeddings, train_sbert_embeddings, valid_sbert_embeddings

def get_image(directory, image_id):
    image = Image.open(os.path.join(directory, image_id))
    return image

def get_image_features(images, image_directory, clip_model, preprocess):
    N = len(images)
    count = 0
    image_features = torch.empty(size=(N, 512))
    for i,image_id in enumerate(images):
        count += 1
        im = get_image(image_directory, image_id)
        image = preprocess(im).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_image = clip_model.encode_image(image)
            image_features[i] = clip_image
    return image_features

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
    fig, axs = plt.subplots(rows, columns, figsize=(15, 7))
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
    if not os.path.exists('imgs'):  # TODO: this should point to a "result" folder out of the project dir
        print("Folder imgs does not exist in current directory")
        os.makedirs('imgs')
        
    path_to_save = os.path.join('imgs', '{}.png'.format(plot_name))
    plt.savefig(path_to_save)
    #plt.show()
    
def get_logits(image_features, text_features):
    # normalized features
    if text_features.dtype == torch.int64:
        text_features = text_features.type(torch.FloatTensor)
    if text_features.dtype == torch.float32:
        text_features = text_features.to(torch.float16)
    if text_features.dtype == torch.float16:
        text_features = text_features.to(torch.float32)
    
    image_features = (image_features / image_features.norm(dim=-1, keepdim=True)).to(device)
    text_features = (text_features / text_features.norm(dim=-1, keepdim=True)).to(device)

    # cosine similarity as logits
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logit_scale = logit_scale.exp().to(device)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text 

def reciprocal_rank(probs, value):
    N = len(probs)
    copy_probs = list(probs.copy())
    for i in range(N):
        max_value = max(copy_probs)
        if max_value == value:
            return 1/(i + 1)
        else:
            copy_probs.remove(max_value)
    return 1/N

def get_MRR(languages, model, sbert_model, clip_model, captions, images_features):
    sbert_lang_performance = []
    clip_lang_performance = []
    sbert_lang_errors = []
    clip_lang_errors = []
    sbert_lang_mrr = []
    clip_lang_mrr = []
    vetoed = []
    for lang, code in languages.items():
        #print("Lang {}".format(lang))
        with torch.no_grad():
            try:
                torch_features = get_sbert_embeddings(captions[lang],sbert_model) 
                sbert_features = model(torch_features.to(device)).type(torch.float16)
                #print("SBERT features ready. Timestamp: {}".format(datetime.now()))
                clip_features = get_clip_embeddings(captions[lang],clip_model).to(device)
                #print("CLIP features ready. Timestamp: {}".format(datetime.now())) 
            except:
                print("Not able to tokenize in {}. Skipping language {}".format(lang, code))
                vetoed.append(lang)
                continue

            sbert_performance = []
            clip_performance = []
            sbert_errors = 0
            clip_errors = 0
            sbert_rr = 0
            clip_rr = 0
            counter = 0
            
            for image_feature in images_features:
                # Get the probabilities for SBERT and CLIP
                logits_image_sbert, logits_text_sbert = get_logits(image_feature, sbert_features)
                logits_image_clip, logits_text_clip = get_logits(image_feature, clip_features)
                probs_clip = logits_image_clip.softmax(dim=-1).to('cpu').numpy()
                probs_sbert = logits_image_sbert.softmax(dim=-1).to('cpu').numpy()


                # Append the probs to array            
                ps = probs_sbert[counter]
                sbert_rr += reciprocal_rank(probs_sbert,ps)
                sbert_performance.append(ps)
                if ps < max(probs_sbert):
                    sbert_errors += 1
                pc = probs_clip[counter]
                clip_rr += reciprocal_rank(probs_clip, pc)
                clip_performance.append(pc)
                if pc < max(probs_clip):
                    clip_errors += 1
                counter += 1

        # print("Images processed: {}".format(counter))
        # print("Classifications errors: SBERT --> {} ; CLIP --> {}".format(sbert_errors,clip_errors))
        sbert_lang_performance.append(round(sum(sbert_performance)/counter,4))
        clip_lang_performance.append(round(sum(clip_performance)/counter,4))
        sbert_lang_mrr.append(round(sbert_rr/counter,3))
        clip_lang_mrr.append(round(clip_rr/counter,3))
        sbert_lang_errors.append(sbert_errors)
        clip_lang_errors.append(clip_errors)
    
    #print("Done")
    #print("Forbidden Languages: {}".format(vetoed))
    #print("SBERT_LANG_PERFORMANCE: {}".format(sbert_lang_performance))
    return sbert_lang_performance, clip_lang_performance, sbert_lang_mrr, clip_lang_mrr, sbert_lang_errors, clip_lang_errors


def train(model, sbert_model, clip_model, train_dataset, valid_dataset, languages, captions, images_features, b_size=32, epochs=200):
    writer = SummaryWriter(comment='train')  # TODO argument log_dir= should be a folder out of the project dir
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5)
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
        wandb.log({"Loss/Train": loss},step=epoch)
        writer.add_scalar("Loss/Train", train_avg_loss, epoch)
        writer.flush()
        
        valid_loss = 0.0
        model.eval()
        valid_counter = 0
        
        if epoch % 50 == 0:
            sbert_per, clip_per, sbert_MRR, clip_MRR, sbert_er, clip_er = get_MRR(languages, model, sbert_model, clip_model, captions, images_features)                 
            #print("SBERT_PER: {}".format(sbert_per))
            for i, (lang, code) in enumerate(languages.items()):
                wandb.log({lang+"/Performance/SBERT": sbert_per[i]}, epoch)
                wandb.log({lang+"/Performance/CLIP": clip_per[i]}, epoch)
                wandb.log({lang+"/MRR/SBERT": sbert_MRR[i]}, epoch)
                wandb.log({lang+"/MRR/CLIP": clip_MRR[i]}, epoch)
                wandb.log({lang+"/Error/SBERT": sbert_er[i]}, epoch)
                wandb.log({lang+"/Error/CLIP": clip_er[i]}, epoch)
                writer.add_scalar(lang+"/Performance/SBERT", sbert_per[i], epoch)
                writer.add_scalar(lang+"/Performance/CLIP", clip_per[i], epoch)
                writer.add_scalar(lang+"/MRR/SBERT", sbert_MRR[i], epoch)
                writer.add_scalar(lang+"/MRR/CLIP", clip_MRR[i], epoch)
                writer.add_scalar(lang+"/Error/SBERT", sbert_er[i], epoch)
                writer.add_scalar(lang+"/Error/CLIP", clip_er[i], epoch)
            
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            target = model(inputs)
            loss = criterion(target.to(float), labels.to(float))
            valid_loss += loss.item()           
            valid_counter += 1
        valid_avg_loss = valid_loss/valid_counter 
        scheduler.step(valid_avg_loss)
        valid_losses.append(valid_avg_loss)
        if epoch % 50 == 0 or epoch == epochs-1:
            new_lr = [ group['lr'] for group in optimizer.param_groups ] 
            print("Epoch {}. Train Loss: {}. Valid Loss: {}. LR: {}".format(epoch, train_loss/train_counter, valid_loss/valid_counter, new_lr[0]))
        
        wandb.log({"Loss/Valid": sbert_MRR[i]}, epoch)
        writer.add_scalar("Loss/Valid", valid_avg_loss, epoch)
        writer.flush()        
    writer.close()
    
    return train_losses, valid_losses

def supra_training(models, sbert_model, clip_model, train_sbert_emb,train_clip_emb, valid_sbert_emb, valid_clip_emb, size, languages, captions, image_features, directory, n_epochs):
    model_train_losses = []
    model_valid_losses = []
    final_loss = []
    training_time = []
    for name, model in models.items():
        start_time = time.time()
        train_dataset = EmbeddingsDataset(train_sbert_emb, train_clip_emb)
        valid_dataset = EmbeddingsDataset(valid_sbert_emb, valid_clip_emb)
        train_loss, valid_loss = train(model, sbert_model, clip_model, train_dataset, valid_dataset, languages, captions, image_features, epochs=n_epochs)
        end_time = time.gmtime(time.time() - start_time)
        elapsed_time = time.strftime("%H:%M:%S", end_time)
        training_time.append(elapsed_time)
        
        model_train_losses.append(train_loss)                   
        model_valid_losses.append(valid_loss)
        final_loss.append(round(valid_loss[-1],5))
        date_time = datetime.now()
        str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
        name_to_save = directory + '_' + name +'_e'+str(n_epochs)+'_s'+str(size)+ '.pt'
        data_json = {"Timestamp":str_date_time, "name_model":name, "hidden_layers":int(name.split("_")[1]), 
                     "epochs":n_epochs, "train_size":size, "train_dataset":directory, "model_saved_as": name_to_save + '.pt',
                     "train_loss":round(train_loss[-1],3), "valid_loss":round(valid_loss[-1],3), "elapsed_time": elapsed_time}        
        with open(os.path.join('jsons',name_to_save+'.json'), 'w', encoding='utf-8') as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)
        logger.info(f'Trained model called {name_to_save} at {str_date_time}')
        if not os.path.exists("models"):  # TODO: this should be out of the project folder
            print("Folder Models does not exist in current directory")
            os.makedirs("models")
        torch.save(model.state_dict(), os.path.join('models', name_to_save))
        print('Finished Training from model {}. Elapsed time: {}.'.format(name,elapsed_time))
        # print("-"*50)
    actual_time = time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(time.time()))
    print("End of Training Process on {}".format(actual_time))
    return model_train_losses, model_valid_losses, training_time, final_loss


def run_pipeline(directory, n_epochs):  # Training Pipeline
    with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    directory = cfg["coco"]["out_dir"]
    image_directory = cfg["coco"]["image_dir"]
    print("Image_directory: {}".format(image_directory))
    languages = cfg["languages"]
    durations = {}
    finals = {}
    sbert_model, clip_model, preprocess = get_sbert_and_clip_models()
    train_clip_emb, valid_clip_emb, train_sbert_emb, valid_sbert_emb = get_train_embeddings(directory,clip_model, sbert_model)
    train_size = train_sbert_emb.size()[0]
    input_size = train_sbert_emb.size()[1]
    model_dict = get_models_to_train(input_size)
    images, captions = get_images_and_captions(directory, languages)
    images_features = get_image_features(images["english"], image_directory, clip_model, preprocess)
    train_losses, valid_losses, train_time, final_loss = supra_training(
        model_dict,
        sbert_model,
        clip_model,
        train_sbert_emb,
        train_clip_emb,
        valid_sbert_emb,
        valid_clip_emb,
        train_size,
        languages,
        captions,
        images_features,
        directory=directory,        
        n_epochs=n_epochs
    )
    durations[directory] = train_time
    finals[directory] = final_loss
    train_final_losses = [x[-1] for x in train_losses]
    train_results = pd.DataFrame({"TrainLoss": train_final_losses, "ValidLoss": final_loss}, index=model_dict.keys())
    print(train_results)
    show_plot(model_dict, train_losses, valid_losses, directory+'_'+str(n_epochs)+'_'+str(train_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCLIP")
    parser.add_argument("--dataset", help="decide which dataset to use", default="europarl", type=str)
    parser.add_argument('--epochs', type=int, default=100, help='how many epoches for this training')
    args = parser.parse_args()

    run_pipeline(args.dataset, args.epochs)
