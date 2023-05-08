import numpy as np
import json
import torch
import torch.nn as nn
from networks import SCLIPNN
import clip
from multilingual_clip import pt_multilingual_clip
import transformers
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torchvision.transforms.functional as fn
import pandas as pd
import yaml
import os
from datetime import datetime
from experiment import *

#######################################
# Models functions
#######################################
def get_sbert_and_clip_and_mclip_models():
    sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    print("SBERT model loaded")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded")
    mclip_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-32')
    mclip_tokenizer = transformers.AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-B-32')
    print("MCLIP model loaded")
    return sbert_model.eval(), (clip_model.eval(), preprocess), (mclip_model.eval(), mclip_tokenizer)

def get_mclip_embeddings(sentences, mclip_model, mclip_tokenizer):
    with torch.no_grad():
        mclip_embeddings = mclip_model.forward(sentences, mclip_tokenizer).to(device)
    return mclip_embeddings

#######################################
# Additional Functions
#######################################
def get_results(sbert_lang_performance, clip_lang_performance, sbert_lang_errors, clip_lang_errors, sbert_lang_mrr, clip_lang_mrr):
    results = pd.DataFrame({"SBERT":sbert_lang_performance, "CLIP": clip_lang_performance,
                        "error SBERT":sbert_lang_errors, "error CLIP":clip_lang_errors,
                       "MRR sbert":sbert_lang_mrr, "MRR clip": clip_lang_mrr}, 
                       index=languages)
    return results

def show_plot(languages, sbert_lang_mrr, clip_lang_mrr, mclip_lang_mrr):
    X_axis = np.arange(len(languages.keys()))
    figure_name = plt.figure(figsize=(20, 8))
    plt.bar(X_axis-0.2, sbert_lang_mrr, 0.4, color = 'blue', edgecolor = 'black', capsize=7, label='SBERT MRR')
    plt.bar(X_axis+0.2, clip_lang_mrr, 0.4, color = 'red', edgecolor = 'black', capsize=7, label='CLIP MRR')
    plt.bar(X_axis+0.2, mclip_lang_mrr, 0.4, color = 'yellow', edgecolor = 'black', capsize=7, label='CLIP MRR')
    plt.xticks(rotation = 45)
    plt.xticks(X_axis, languages.keys())
    plt.legend()
    plt.show()
    
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

def get_MRR(languages, name_of_model, sbert_model, clip_model, mclip_model, mclip_tokenizer, captions, images_features):
    sbert_lang_performance = []
    clip_lang_performance = []
    mclip_lang_performance = []
    sbert_lang_errors = []
    clip_lang_errors = []
    mclip_lang_errors = []
    sbert_lang_mrr = []
    clip_lang_mrr = []
    mclip_lang_mrr = []
    vetoed = []
    for lang, code in languages.items():
        print("Processing captions in "+ lang +"...")

        with torch.no_grad():
            try:
                torch_features = get_sbert_embeddings(captions[lang],sbert_model) 
                sbert_features = sbert_to_clip(torch_features,name_of_model).type(torch.float16)
                print("SBERT features ready. Timestamp: {}".format(datetime.now()))
                clip_features = get_clip_embeddings(captions[lang],clip_model).to(device)
                print("CLIP features ready. Timestamp: {}".format(datetime.now()))
                mclip_features = get_mclip_embeddings(captions[lang],mclip_model,mclip_tokenizer)
                print("MCLIP features ready. Timestamp: {}".format(datetime.now()))
            except:
                print("Not able to tokenize in {}. Skipping language {}".format(lang, code))
                vetoed.append(lang)
                continue
            
        print("Encodings complete")

        sbert_performance = []
        clip_performance = []
        mclip_performance = []
        sbert_errors = 0
        clip_errors = 0
        mclip_errors = 0
        sbert_rr = 0
        clip_rr = 0
        mclip_rr = 0
        counter = 0
        
        for image_feature in images_features:
            # Get the probabilities for SBERT and CLIP
            logits_image_sbert, logits_text_sbert = get_logits(image_feature, sbert_features,'sbert')
            logits_image_mclip, logits_text_mclip = get_logits(image_feature, mclip_features,'mclip')
            logits_image_clip, logits_text_clip = get_logits(image_feature, clip_features,'clip')
            probs_sbert = logits_image_sbert.softmax(dim=-1).cpu().detach().numpy()
            probs_mclip = logits_image_mclip.softmax(dim=-1).cpu().detach().numpy()
            probs_clip = logits_image_clip.softmax(dim=-1).cpu().detach().numpy()
            
            # Append the probs to array            
            ps = probs_sbert[counter]
            sbert_rr += reciprocal_rank(probs_sbert,ps)
            sbert_performance.append(ps)
            if ps < max(probs_sbert):
                sbert_errors += 1
            pc = probs_clip[counter]
            clip_rr += reciprocal_rank(probs_clip,pc)
            clip_performance.append(pc)
            if pc < max(probs_clip):
                clip_errors += 1
            pm = probs_mclip[counter]
            mclip_rr += reciprocal_rank(probs_mclip,pm)
            mclip_performance.append(pm)
            if pm < max(probs_mclip):
                mclip_errors += 1
            counter += 1

            if counter % 100 == 0:
                print("{} images already processed for {}".format(counter,lang))

        print('-'*70)
        # print("Images processed: {}".format(counter))
        # print("Classifications errors: SBERT --> {} ; CLIP --> {}".format(sbert_errors,clip_errors))
        sbert_lang_performance.append(round(sum(sbert_performance)/counter,6))
        clip_lang_performance.append(round(sum(clip_performance)/counter,6))
        mclip_lang_performance.append(round(sum(mclip_performance)/counter,6))
        sbert_lang_mrr.append(round(sbert_rr/counter,4))
        clip_lang_mrr.append(round(clip_rr/counter,4))
        mclip_lang_mrr.append(round(mclip_rr/counter,4))
        sbert_lang_errors.append(sbert_errors)
        clip_lang_errors.append(clip_errors)
        mclip_lang_errors.append(mclip_errors)
    print("Done")
    print("Forbidden Languages: {}".format(vetoed))
    
    return sbert_lang_performance, clip_lang_performance, mclip_lang_performance, sbert_lang_mrr, clip_lang_mrr, mclip_lang_mrr, sbert_lang_errors, clip_lang_errors, mclip_lang_errors

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    directory = cfg["coco"]["out_dir"]
    image_directory = cfg["coco"]["image_dir"]
    print("Image_directory: {}".format(image_directory))
    languages = cfg["languages"]
    model_dir = cfg["models"]["model_dir"]
    name_of_model = 'coco_NN_900_e300_s400000.pt'
    trained_model = os.path.join(model_dir,name_of_model)
    print(languages)
    sbert_model, (clip_model, preprocess), (mclip_model, mclip_tokenizer) = get_sbert_and_clip_and_mclip_models()