import numpy as np
import json
import torch
import torch.nn as nn
from networks import SCLIPNN, SCLIPNN3
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import torchvision.transforms.functional as fn
import pandas as pd
import yaml
import os
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
directory = cfg["coco"]["out_dir"]
image_directory = cfg["coco"]["image_dir"]
languages = cfg["languages"]

#######################################
# Models functions
#######################################
def get_sbert_and_clip_models():
    sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("SBERT model loaded")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded")
    return sbert_model.eval(), clip_model.eval(), preprocess

def get_sbert_embeddings(sentences, sbert_model):
    with torch.no_grad():  
        sbert_embeddings = torch.from_numpy(sbert_model.encode(sentences))
    return sbert_embeddings

def get_clip_embeddings(sentences, clip_model, batch_size=10):
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

def load_model(path_to_model, sbert_model, hidden_size=1000):
    PATH = path_to_model
    sbert_features = get_sbert_embeddings(['simple sentence'],sbert_model)
    input_size = sbert_features.shape[1]
    if 'NN3' in path_to_model:
        model = SCLIPNN3(input_size, hidden_size)
        model.load_state_dict(torch.load(PATH))
    else:
        model = SCLIPNN(input_size, hidden_size)
        model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def sbert_to_clip(sbert_features, name_model):
    splitted_name = name_model.split("_")
    hidden_size = int(splitted_name[2])
    input_size = sbert_features.shape[1]
    PATH = os.path.join("models",name_model)
    if 'NN3' in name_model:
        model = SCLIPNN3(input_size,hidden_size)
    else:
        model = SCLIPNN(input_size,hidden_size)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    output = model(sbert_features)
    return output

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

#################################
# Images functions
#################################
def get_image(directory, image_id):
    image = Image.open(os.path.join(directory, image_id))
    return image

def reshape(im):
    print("This is size of original image:",im.size, "\n")
    width, height = im.size
    # print("W: {} and H: {}".format(width, height))
    if width > 1000 or height > 1000:
        scale = 3
    elif width > 500 or height > 500:
        scale = 2
    else:
        scale = 1    
    new_width = int(width / scale)
    new_height = int(height / scale)
    #image = preprocess(im)
    image = fn.resize(im, size=[new_width])
    print("This is size of resized image:",image.size, "\n")
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

def get_image_and_captions_clip_features(directory, languages, image_directory,clip_model, preprocess):
    images, captions = get_images_and_captions(directory, languages)
    images_features = {}
    clip_features = {}
    for lang in languages.keys():
        images_features[lang] = get_image_features(images[lang],image_directory,clip_model, preprocess)
        clip_features[lang] = get_clip_features(captions[lang],clip_model).to(device)
    return images_features, clip_features, captions

#######################################
# Additional Functions
#######################################
def get_results(sbert_lang_performance, clip_lang_performance, sbert_lang_mrr, clip_lang_mrr, sbert_lang_errors, clip_lang_errors, sbert_lang_correct, clip_lang_correct):
    results = pd.DataFrame({"SBERT":sbert_lang_performance, "CLIP": clip_lang_performance,
                            "MRR sbert":sbert_lang_mrr, "MRR clip": clip_lang_mrr,
                        "Error SBERT":sbert_lang_errors, "Error CLIP":clip_lang_errors,
                            "Accuracy SBERT":sbert_lang_correct, "Accuracy CLIP":clip_lang_correct
                       }, 
                       index=languages)
    #print(results)
    return results


def reciprocal_rank(probs, value):
    return float(1 / (1 + np.where(-np.sort(-probs) == value)[0][0]))

def reciprocal_rank_b(probs, value):
    N = len(probs)
    copy_probs = list(probs.copy())
    for i in range(N):
        max_value = max(copy_probs)
        if max_value == value:
            return 1/(i + 1)
        else:
            copy_probs.remove(max_value)
    return 1/N


def get_MRR(languages, model, name_of_model, sbert_model, clip_model, captions, images_features):
    sbert_lang_performance = []
    clip_lang_performance = []
    sbert_lang_errors = []
    clip_lang_errors = []
    sbert_lang_correct = []
    clip_lang_correct = []
    sbert_lang_mrr = []
    clip_lang_mrr = []
    vetoed = []
    for lang, code in languages.items():
        print("Lang {}. Timestamp: {}".format(lang, datetime.now()))
        with torch.no_grad():
            try:
                torch_features = get_sbert_embeddings(captions[lang],sbert_model) 
                sbert_features = sbert_to_clip(torch_features,name_of_model).type(torch.float16)
                #print("SBERT features ready. Timestamp: {}".format(datetime.now()))
            except:
                print("[SBERT] Not able to extract features in {}. Skipping language {}".format(lang, code))
            try:
                clip_features = get_clip_embeddings(captions[lang],clip_model).to(device)
                #print("CLIP features ready. Timestamp: {}".format(datetime.now())) 
            except:
                print("[CLIP] Not able to tokenize in {}. Skipping language {}".format(lang, code))
                vetoed.append(lang)
                #continue

            sbert_performance = []
            clip_performance = []
            sbert_errors = 0
            clip_errors = 0
            sbert_correct = 0
            clip_correct = 0
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
                elif ps == max(probs_sbert):
                    sbert_correct +=1
                pc = probs_clip[counter]
                clip_rr += reciprocal_rank(probs_clip, pc)
                clip_performance.append(pc)
                if pc < max(probs_clip):
                    clip_errors += 1
                elif pc == max(probs_clip):
                    clip_correct += 1
                counter += 1

        # print("Images processed: {}".format(counter))
        # print("Classifications errors: SBERT --> {} ; CLIP --> {}".format(sbert_errors,clip_errors))
        sbert_lang_performance.append(round(sum(sbert_performance)/counter,4))
        clip_lang_performance.append(round(sum(clip_performance)/counter,4))
        sbert_lang_mrr.append(round(sbert_rr/counter,3))
        clip_lang_mrr.append(round(clip_rr/counter,3))
        sbert_lang_errors.append(sbert_errors/counter)
        clip_lang_errors.append(clip_errors/counter)
        sbert_lang_correct.append(sbert_correct/counter)
        clip_lang_correct.append(clip_correct/counter)
        
    
    #print("Done")
    #print("Forbidden Languages: {}".format(vetoed))
    #print("SBERT_LANG_PERFORMANCE: {}".format(sbert_lang_performance))
    return sbert_lang_performance, clip_lang_performance, sbert_lang_mrr, clip_lang_mrr, sbert_lang_errors, clip_lang_errors, sbert_lang_correct, clip_lang_correct

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    directory = cfg["coco"]["out_dir"]
    image_directory = cfg["coco"]["image_dir"]
    languages = cfg["languages"]
    model_dir = cfg["models"]["model_dir"]
    name_of_model = 'coco_NN_900_e300_s400000.pt'
    trained_model = os.path.join(model_dir,name_of_model)
    sbert_model, clip_model, preprocess = get_sbert_and_clip_models()
    images, captions = get_images_and_captions(languages)
    images_features = get_image_features(images["english"], image_directory, clip_model, preprocess)
    model = load_model(trained_model,sbert_model)
    sbert_per, clip_per, sbert_MRR, clip_MRR, sbert_errors, clip_errors = get_MRR(languages,model,name_of_model,sbert_model,clip_model,captions, images_features)
    display_results(sbert_per,clip_per,sbert_errors, clip_errors,sbert_MRR,clip_MRR) 