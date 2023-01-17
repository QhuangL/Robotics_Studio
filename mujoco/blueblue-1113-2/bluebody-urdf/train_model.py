import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import time
import os


class pred_data(Dataset):
    def __init__(self, data):
        self.input_data = data[:, :3]  # pos
        self.label_data = data[:, 3:]  # angle
        ## normalize to 0-1
        self.label_data[:, 0] = (self.label_data[:, 0] + 1.) / 2.
        self.label_data[:, 1] = (self.label_data[:, 1] + .5) / 1.5
        self.label_data[:, 2] = (self.label_data[:, 2] + 1.) / 2.

    def __getitem__(self, idx):

        input_data_sample = self.input_data[idx]
        label_data_sample = self.label_data[idx]
        input_data_sample = torch.from_numpy(input_data_sample).to(device, dtype=torch.float)
        label_data_sample = torch.from_numpy(label_data_sample).to(device, dtype=torch.float)
        sample = {"input": input_data_sample, "label": label_data_sample}
        return sample

    def __len__(self):
        return len(self.input_data)

class IKModel(nn.Module):
    def __init__(self):
        super(IKModel, self).__init__()
        self.input_size = 3
        self.output_size = 3

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return torch.sigmoid(self.fc3(x))


def train(model, batchsize, lr, num_epoch, log_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    Loss_fun = nn.MSELoss(reduction='mean')
    train_dataset = pred_data(data = training_d)
    test_dataset = pred_data(data = test_d)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    train_epoch_L = []
    test_epoch_L  = []
    min_loss = + np.inf
    for epoch in range(num_epoch):
        t0 = time.time()
        model.train()
        temp_l = []
        running_loss = 0.0
        for i, bundle in enumerate(train_dataloader):
            input_d, label_d = bundle["input"], bundle["label"]

            input_d = torch.flatten(input_d, 1)
            label_d = torch.flatten(label_d, 1)

            pred_result = model.forward(input_d)

            # loss = model.loss(pred_result,label_d)
            loss = Loss_fun(pred_result, label_d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            temp_l.append(loss.item())
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 200 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
            #     running_loss = 0.0

        train_mean_loss = np.mean(temp_l)
        train_epoch_L.append(train_mean_loss)

        model.eval()
        temp_l = []

        with torch.no_grad():
            for i, bundle in enumerate(test_dataloader):
                input_d, label_d = bundle["input"], bundle["label"]

                input_d = torch.flatten(input_d, 1)
                label_d = torch.flatten(label_d, 1)
                pred_result = model.forward(input_d)
                # loss = model.loss(pred_result, label_d)
                loss = Loss_fun(pred_result, label_d)
                temp_l.append(loss.item())

            test_mean_loss = np.mean(temp_l)
            test_epoch_L.append(test_mean_loss)

        if test_mean_loss < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(train_mean_loss))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(test_mean_loss))
            min_loss = test_mean_loss
            PATH = log_path + '/best_model_MSE.pt'
            torch.save(model.state_dict(), PATH)
        np.savetxt(log_path + "training_MSE.csv", np.asarray(train_epoch_L))
        np.savetxt(log_path + "testing_MSE.csv", np.asarray(test_epoch_L))

        t1 = time.time()
        # print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "training mean loss: ",train_mean_loss, "lr:", lr)
        print(epoch, "training loss: ",train_mean_loss, "Test loss: ", test_mean_loss)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("start", device)

    data = np.loadtxt("data/ik_data_10000.csv")
    np.random.shuffle(data)
    training_d, test_d = data[:9000,:], data[9000:,:]

    # DATA_PATH = "./data"

    Lr = 1e-4
    Batch_size = 12  # 128
    Num_epoch = 3000

    Log_path = "./log_03/"

    try:
        os.mkdir(Log_path)
    except OSError:
        pass
    Model = IKModel().to(device)

    train(model=Model, batchsize=Batch_size, lr=Lr, num_epoch=Num_epoch, log_path=Log_path)


