import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from atv_model import AtvModel
from excavator_model import ExcavatorModel
from trailer_model import TrailerModel

device = torch.device("cuda:0")
lr = 0.01
epochs = 50000

class TsvDataset:
    def __init__(self, filename: str, label_column: int):

        # Load data
        np_data = np.genfromtxt(fname=filename, skip_header=1, delimiter="\t", filling_values=0)
        self.data = torch.from_numpy(np_data).to(device)
        self.labels = self.data[:, label_column]
        self.pd_data = pd.read_csv(filename, sep="\t")

    def get_labels(self):
        return self.labels

subject = 'atv' # 'excavator', 'atv'

# dataset = TsvDataset("data/excavator.tsv", 15)
# model = ExcavatorModel(DEVICE, dataset.pd_data['Make'], 'config/excavator/config_8343.json')
# dataset = TsvDataset(f"data/trailer.tsv", 0)
# model = TrailerModel(f'config/trailer/config_{config_num}.json')
dataset = TsvDataset("data/atv.tsv", 0)
model = AtvModel('config/atv/config_811.json')
model.to(device=device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
best_loss = model.fit(optimizer, scheduler, dataset.data, dataset.labels, epochs)

model.save_params(f"config/{subject}", best_loss)
