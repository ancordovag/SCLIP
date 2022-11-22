from preprocessing.preprocessing import write_files
from preprocessing.train_mixer import mix_files
from sclip_training import run_pipeline
from bertin_training import run_bertin_pipeline
import yaml
import os

# MIX should be at the end
source_dirs = ['esco']
if 'mix' in source_dirs:
    if 'europarl' not in source_dirs or 'coco' not in source_dirs:
        print("Cannot mix if there is no europarl or coco datasets to process")
        source_dirs.remove('mix')
train_sizes = [x*100000 for x in range(1,6)]
number_epochs = [300]
with open(os.path.join('preprocessing','config.yml'), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

for tsz in train_sizes:
    val_test_size = int(tsz/4)
    for i, train_dir in enumerate(source_dirs):
        if 'mix' in train_dir:
            params = cfg[train_dir]
            mix_files(params["europarl_dir"], params["coco_dir"], params["out_dir"])            
        else:
            params = cfg[train_dir]
            data_dir = params['data_dir']
            out_dir = params['out_dir'] 
            try:
                write_files(train_dir,tsz, val_test_size, data_dir=data_dir, out_dir=out_dir)
            except:
                print(f'Directory {source_dir} does not exist. Moving on.')
                continue
        for epochs in number_epochs:
            print("---"*20)
            print(f'Training with trainset {train_dir}, with {tsz} sentences, {epochs} epochs')
            if train_dir == 'esco':
                run_bertin_pipeline(train_dir,epochs)
            else:
                run_pipeline(train_dir, epochs)

print("Meta Runner Finished")