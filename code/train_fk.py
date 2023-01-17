import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
import torchvision
from model import FKModel
import numpy as np
import os
import time

# Check GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("start", device)

class PredData(Dataset):
    def __init__(self, data_len):
        self.len = data_len

    def __getitem__(self, idx):

        input_data_sample = np.random.rand(3)
        label_data_sample = np.array([0.])
        input_data_sample = torch.from_numpy(input_data_sample).to(device, dtype=torch.float)
        label_data_sample = torch.from_numpy(label_data_sample).to(device, dtype=torch.float)
        sample = {"input": input_data_sample, "label": label_data_sample}
        return sample

    def __len__(self):
        return self.len

def train_model(model, batch_size, lr, num_epoch, log_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = PredData(10000)
    test_dataset = PredData(2000)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    train_epoch_L = []
    test_epoch_L = []
    min_loss = + np.inf

    for epoch in range(num_epoch):
        t0 = time.time()
        model.train()
        temp_l = []
        running_loss = 0.0
        for i, bundle in enumerate(train_dataloader):
            input_d, _ = bundle["input"], bundle["label"]

            input_d = torch.flatten(input_d, 1)
            # label_d = torch.flatten(label_d, 1)

            pred_result = model.forward(input_d)

            loss = model.loss(input_d, pred_result)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_l.append(loss.item())
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
                running_loss = 0.0

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        if train_mean_loss < min_loss:
            min_loss = train_mean_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        print(epoch, "training loss: ", train_mean_loss)


if __name__ == "__main__":
    Batch_size = 2
    Lr = 1e-4
    Num_epoch = 1000000

    Log_path = "./log_fk_model/"

    try:
        os.mkdir(Log_path)
    except OSError:
        pass

    Model = FKModel(bsize=Batch_size).to(device)
    train_model(model=Model, batch_size=Batch_size, lr=Lr, num_epoch=Num_epoch, log_path=Log_path)
