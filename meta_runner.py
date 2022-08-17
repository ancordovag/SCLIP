from preprocessing.europarl_preprocessing import write_new_files
from preprocessing.coco_preprocessing import write_coco_files
from sclip_training import run_pipeline

data_dirs = ['europarl']
source_dirs = ['europarl']
train_sizes = [90000]
number_epochs = [250]

for tsz in train_sizes:
    val_test_size = int(tsz/4)
    for i, source_dir in enumerate(source_dirs):
        if source_dir == 'europarl':
            write_new_files(tsz, val_test_size, val_test_size, data_dir=data_dirs[i], out_dir=source_dir)
        elif source_dir == 'coco':
            write_coco_files(tsz, val_test_size, val_test_size)
        else:
            print(f'Directory {source_dir} does not exist. Moving on.')
            continue     
        for epochs in number_epochs:
            print("---"*20)
            print(f'Training with trainset {source_dir}, with {tsz} sentences, {epochs} epochs')
            run_pipeline(source_dir, epochs)