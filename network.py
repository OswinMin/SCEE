import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
from Simset import *
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class DK(nn.Module):
    def __init__(self, num, inputsize):
        super(DK, self).__init__()
        self.num = num
        self.inputsize = inputsize
        self.cod_fc1 = nn.Linear(inputsize, self.num[0])
        self.cod_fc2 = nn.Linear(self.num[0], self.num[1])
        self.decod_fc1 = nn.Linear(self.num[1], self.num[0])
        self.decod_fc2 = nn.Linear(self.num[0], self.inputsize)
        self.turn_fc1 = nn.Linear(self.num[1], self.num[0])
        self.turn_fc2 = nn.Linear(self.num[0], self.inputsize)
        self.cl_fc1 = nn.Linear(self.inputsize, self.num[0])
        self.cl_fc2 = nn.Linear(self.num[0], self.num[1])
        self.cl_fc3 = nn.Linear(self.num[1], 1)
        self.Lrelu = nn.LeakyReLU(1e-2)
        self.tanh = nn.Tanh()
        self.cri = nn.MSELoss()
        self.opt = optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.Lrelu(self.cod_fc2(self.Lrelu(self.cod_fc1(x))))
        y = self.Lrelu(self.turn_fc2(self.Lrelu(self.turn_fc1(x))))
        x = self.Lrelu(self.decod_fc2(self.Lrelu(self.decod_fc1(x))))
        label = self.tanh(self.cl_fc3(self.Lrelu(self.cl_fc2(self.Lrelu(self.cl_fc1(y))))))
        return x, y, label # decoding output, target output

    def forward1(self, X, targ):
        x = self.Lrelu(self.cod_fc2(self.Lrelu(self.cod_fc1(X))))
        y = self.Lrelu(self.turn_fc2(self.Lrelu(self.turn_fc1(x))))
        y[targ==-1, :] = X[targ==-1, :]
        label = self.tanh(self.cl_fc3(self.Lrelu(self.cl_fc2(self.Lrelu(self.cl_fc1(y))))))
        return label

    def LoadData(self, data):
        self.data = data

    def pretrain(self, epoch):
        params_up = []
        for name, param in self.named_parameters():
            if 'cod' in name or 'decod' in name:
                params_up.append(param)
        optimizer = optim.SGD(params_up, lr=0.01)

        for ep in range(epoch):
            l = 0
            i = 0
            for X, targ in self.data:
                output, _, _ = self.forward(X)
                self.opt.zero_grad()
                loss = self.cri(X[targ==1, :], output[targ==1, :])
                loss.backward()
                optimizer.step()
                l += loss.item()
                i += 1
            print(f"Epoch {ep}, last MSE {round(l/i, 4)}")

    def train_cl(self, epoch):
        params_up = []
        for name, param in self.named_parameters():
            if 'cl' in name:
                params_up.append(param)
        optimizer = optim.SGD(params_up, lr=0.01)
        for ep in range(epoch):
            for X, targ in self.data:
                _, _, label = self.forward(X)
                # label = self.forward1(X, targ)
                self.opt.zero_grad()
                loss = self.cri(label.reshape(-1), targ)
                loss.backward()
                optimizer.step()

    def train(self, epoch, pa):
        self.pretrain(50)
        params_up = []
        for name, param in self.named_parameters():
            if 'cl' not in name:
                params_up.append(param)
        optimizer = optim.SGD(params_up, lr=0.01)
        for ep in range(epoch):
            self.train_cl(10)
            l1 = 0
            l2 = 0
            i = 0
            for X, targ in self.data:
                self.opt.zero_grad()
                _, y, label = self.forward(X)
                trueX = X[targ==1, :]
                trueXy = y[targ==1, :]
                loss1 = self.cri(trueX, trueXy) / pa
                loss2 = -self.cri(label[targ==1, :].reshape(-1), targ[targ==1])
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                l1 += loss1.item()
                l2 += loss2.item()
                i += 1
            print(f"epoch {ep}, loss1 {round(l1/i, 4)}, loss2 {round(l2/i, 4)}")

def umap_draw(result):
    umap1 = umap.UMAP(n_components=2)
    X_umap = umap1.fit_transform(result[:,:-1])
    umap_data = pd.DataFrame(np.concatenate([X_umap, result[:,-1].reshape(-1,1)], axis=1), columns=['Dim1', 'Dim2', 'batch'])
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=umap_data, hue='batch', x='Dim1', y='Dim2')
    plt.title('Umap Visualization of features')
    plt.legend(loc='best')
    # plt.savefig('Umap visualization of features.jpg')
    plt.show()

if __name__ == '__main__':
    sim = gen(5000, 50)
    sm = Simdata(sim)
    data = DataLoader(sm, 32, shuffle=True)
    dk = DK([32, 16], 50)
    dk.LoadData(data)
    dk.train(70, 5)

    _, resultX, _ = dk.forward(sim[0])
    _, resultY, _ = dk.forward(sim[1])
    result = np.concatenate([resultX.detach().numpy(), resultY.detach().numpy()], axis=0)
    batch = np.concatenate((np.ones(5000), -np.ones(5000))).reshape(-1, 1)
    result = np.concatenate([result, batch], axis=1)
    umap_draw(result)