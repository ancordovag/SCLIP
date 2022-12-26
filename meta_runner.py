from preprocessing.preprocessing import write_files
from preprocessing.train_mixer import mix_files
from sclip_training import run_pipeline
from bertin_training import run_bertin_pipeline
import yaml
import os

source_dirs = ['coco']
train_sizes = [410000]
number_epochs = [300]
with open(os.path.join('preprocessing','config.yml'), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

for tsz in train_sizes:
    val_test_size = int(tsz/4)
    for i, train_dir in enumerate(source_dirs):
        params = cfg[train_dir]
        data_dir = params['data_dir']
        out_dir = params['out_dir'] 
        try:
            write_files(train_dir,tsz, val_test_size, data_dir=data_dir, out_dir=out_dir)
            if train_dir == 'esco':
                write_files(train_dir,tsz, val_test_size, data_dir='coco', out_dir='coco')
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