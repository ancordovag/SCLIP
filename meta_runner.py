from europarl_preprocessing import write_new_files
from coco_preprocessing import write_coco_files
from sclip_training import run_pipeline
import sys

directories = ['europarl']
train_sizes = [x*5000 for x in range(1,3)]
number_epochs = [250]

for tsz in train_sizes:
    val_test_size = int(tsz/4)
    for directory in directories:  
        if directory == 'europarl':
            write_new_files(tsz, val_test_size,val_test_size)
        elif directory == 'coco':
            write_coco_files(tsz, val_test_size, val_test_size)
        else:
            print(f'Directory {directory} does not exist. Moving on.')
            continue     
        for epochs in number_epochs:
            print("---"*20)
            print(f'Training with trainset {directory}, with {tsz} sentences, {epochs} epochs')
            run_pipeline(directory,epochs)