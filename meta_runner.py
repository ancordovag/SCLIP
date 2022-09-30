from preprocessing.preprocessing import write_files
from sclip_training import run_pipeline
import yaml
import os

source_dirs = ['europarl','coco']
train_sizes = [x*200 for x in range(1,5)]
number_epochs = [250]
with open(os.path.join('preprocessing','config.yml'), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

for tsz in train_sizes:
    val_test_size = int(tsz/4)
    for i, train_dir in enumerate(source_dirs):
        params = cfg[train_dir]
        data_dir = params['data_dir']
        out_dir = params['out_dir']        
        #try:
        write_files(train_dir,tsz, val_test_size, data_dir=data_dir, out_dir=out_dir)
        #except:
        #    print(f'Directory {source_dir} does not exist. Moving on.')
        #    continue
        for epochs in number_epochs:
            print("---"*20)
            print(f'Training with trainset {train_dir}, with {tsz} sentences, {epochs} epochs')
            run_pipeline(train_dir, epochs)

print("Meta Runner Finished")