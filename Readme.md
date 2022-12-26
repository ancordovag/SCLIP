# SCLIP: SBERT + CLIP
## Introduction

This project mixes SBERT and CLIP to get better results out of CLIP in a multi-language setup.

## 0. Requirements
Install dependences from ```requirements.txt```. <br>
Download MSCOCO dataset in the folder ```SCLIP/coco```. The annotations were gotten from [MSCOCO 2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and the images from [MSCOCO 2017 Train Images](http://images.cocodataset.org/zips/train2017.zip) and [MSCOCO 2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip). 
Download only the annotations of [Google Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download) (Image Labels). 

Edit the path in `preprocessing/config.yaml` to point to your dataset folder (e.g. _/data/SCLIP_).

## 1. Preprocessing
1.1 To generate train, validation and test files for each dataset, run coco_breaker.py and gcc_breaker.py
1.2 For testing, a list of pairs Image-caption is needed. For that, run pairs.py
1.3 As the testing is needed for several languages, the above mentioned pairs list should be translated by running translate.py

## 2. Train
Train SCLIP with meta_runner.py for training with different epochs and train sizes. To see plots, meta_runner.ipynb is also available.
Train SCLIP only with scrip_training.py for just one run with fixed epochs and train size.

## 3. Evaluate
To compare the performance of the different models the experiment.py (also a notebook version available) show the Mean Reciprocal Rank of the trained model over SBERT trying to guess wich caption belong to each image, and do the same with CLIP. For a comparison with Multilingual CLIP realeased this year, experiment-three.ipynb can be excecuted.  
