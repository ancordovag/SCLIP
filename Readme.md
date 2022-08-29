# SCLIP: SBERT + CLIP
## Introduction

This project mixes SBERT and CLIP to get better results out of CLIP in a multi-language setup.

## Requirements
Install dependences from ```requirements.txt```. <br>
Download the Europarl corpus and COCO in your dataset folder (e.g. _/data/SCLIP_):

```
./get_data /data/SCLIP
```

Edit the path in `preprocessing/config.yaml` to point to your dataset folder (e.g. _/data/SCLIP_).
Finally Generate all the training files by runnning: 

```sh
cd preprocessing && ./preprocessing.sh
```

## Usage

Train SCLIP with: 

```sh
python sclip_training.py --dataset europarl --epochs 100
```

or using the ```meta_runner.py```