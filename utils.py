import torch
from torch.utils.data import Dataset
from networks import SCLIPNN, SCLIPNN3

class EmbeddingsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        item_x = self.X[index]
        item_y = self.Y[index]
        return item_x, item_y
    
def get_models_to_train(input_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Creating Models to train...")
    NN_900 = SCLIPNN(input_size, 900).to(device)
    models = {'NN_900':NN_900}
    if len(models) == 1:
        print(f'1 model created : {models.keys()}')
    else:
        print(f'{len(models)} models created.')
    return models