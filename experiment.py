import numpy as np
import json
import torch
import torch.nn as nn
from networks import SCLIPNN
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import torchvision.transforms.functional as fn
import pandas as pd
import yaml
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(os.path.join("preprocessing", "config.yml"), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
directory = cfg["coco"]["out_dir"]
image_directory = cfg["coco"]["image_dir"]
languages = cfg["languages"]


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


def get_sbert_and_clip_models():
    sbert_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return sbert_model.eval(), clip_model.eval(), preprocess


def get_sbert_embeddings(sentences, sbert_model):
    with torch.no_grad():  
        sbert_embeddings = torch.from_numpy(sbert_model.encode(sentences))
    return sbert_embeddings


def load_model(path_to_model,sbert_model):
    PATH = path_to_mode
    sbert_features = get_sbert_embeddings(['simple sentence'],sbert_model)
    input_size = sbert_features.shape[1]
    model = SCLIPNN(input_size,900)
    model.load_state_dict(torch.load(PATH))


def get_logits(image_features, text_features):
    # normalized features
    if text_features.dtype == torch.int64:
        text_features = text_features.type(torch.FloatTensor)
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
    return float(1 / (1 + np.where(-np.sort(-probs) == value)[0][0]))


def get_images_and_captions(languages):
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


def get_image_features(images, image_directory, clip_model, preprocess):
    image_features = []
    #count = 0
    for image_id in images:
    #    count += 1
        im = get_image(image_directory, image_id)
        image = preprocess(im).unsqueeze(0)#.to(device)
    #    if count % 125 == 0:
    #        print(f'DEBUG EXPERIMENT. Count: {count}, image_id: {image_id}')
    #        print(f'Types: image_id {type(image_id)}, im: {type(im)}, image: {type(image)}')
    #        print(f'IM: {im.size}.')
    #        print(f'Image: {image.size()}')
        image_features.append(clip_model.encode_image(image).cpu())
    return image_features   


def get_clip_features(captions, clip_model):
    with torch.no_grad():
        tokenized_features = clip.tokenize(captions).to(device)
        clip_features = clip_model.encode_text(tokenized_features)
    return clip_features  


def get_MRR(model,directory, languages,sbert_model,captions,images_features, clip_features):
    sbert_lang_performance = []
    clip_lang_performance = []
    sbert_lang_errors = []
    clip_lang_errors = []
    sbert_lang_mrr = []
    clip_lang_mrr = []
    vetoed = []
    for lang, code in languages.items():

        with torch.no_grad():
            try:
                torch_features = torch.from_numpy(sbert_model.encode(captions[lang])).to(device)
                sbert_features = model(torch_features).type(torch.float16)                
            except:
                #print("Not able to tokenize in {}. Skipping language {}".format(lang, code))
                vetoed.append(lang)
                continue

            sbert_performance = []
            clip_performance = []
            sbert_errors = 0
            clip_errors = 0
            sbert_rr = 0
            clip_rr = 0
            counter = 0

            for image_feature in images_features[lang]:
                # Get the probabilities for SBERT and CLIP
                logits_image_sbert, logits_text_sbert = get_logits(image_feature, sbert_features)
                logits_image_clip, logits_text_clip = get_logits(image_feature, clip_features[lang])
                probs_clip = logits_image_clip.softmax(dim=-1).cpu().numpy()
                probs_sbert = logits_image_sbert.softmax(dim=-1).cpu().numpy()

                # Append the probs to array            
                ps = probs_sbert[0][counter]
                sbert_rr += reciprocal_rank(probs_sbert[0],ps)
                sbert_performance.append(ps)
                if ps < max(probs_sbert[0]):
                    sbert_errors += 1
                pc = probs_clip[0][counter]
                clip_rr += reciprocal_rank(probs_clip[0], pc)
                clip_performance.append(pc)
                if pc < max(probs_clip[0]):
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
    return sbert_lang_performance, clip_lang_performance, sbert_lang_mrr, clip_lang_mrr, sbert_lang_errors, clip_lang_errors


def display_results(sbert_lang_performance, clip_lang_performance, sbert_lang_errors, clip_lang_errors, sbert_lang_mrr, clip_lang_mrr):
    results = pd.DataFrame({"SBERT":sbert_lang_performance, "CLIP": clip_lang_performance,
                        "error SBERT":sbert_lang_errors, "error CLIP":clip_lang_errors,
                       "MRR sbert":sbert_lang_mrr, "MRR clip": clip_lang_mrr}, 
                       index=languages)
    print(results)


def get_image_and_captions_clip_features(languages, image_directory,clip_model, preprocess):
    images, captions = get_images_and_captions(languages)
    images_features = {}
    clip_features = {}
    for lang in languages.keys():
        images_features[lang] = get_image_features(images[lang],image_directory,clip_model, preprocess)
        clip_features[lang] = get_clip_features(captions[lang],clip_model).cpu()
    return images_features, clip_features, captions


if __name__ == "__main__":
    sbert_model, clip_model, preprocess = get_sbert_and_clip_models()
    images_features, clip_features, captions = get_image_and_captions_clip_features(languages, image_directory,clip_model, preprocess)
    model = load_model('models/best_model.pt'),
    sbert_per, clip_per, sbert_MRR, clip_MRR, sbert_errors, clip_errors = get_MRR(model,directory, languages,sbert_model,captions, images_features,clip_features)
    display_results(sbert_per,clip_per,sbert_errors, clip_errors,sbert_MRR,clip_MRR) 