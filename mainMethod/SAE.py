import pandas as pd
import scanpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import SGD
import numpy as np
from Cellset import Cellset
import scanpy as sc
import os


class SAE(nn.Module):
    def __init__(self, dims, epochs_per_layer, lr=0.05, batch_size=32, save_dir='',
                 lossF='MSE'):
        """
        dims：一个列表，表示每一层的输入输出维度，例如[784, 256, 128, 64]表示有3层编码器
        epochs_per_layer：每一层编码器的训练轮数
        lr：学习率
        batch_size：每个训练循环中，使用DataLoader加载训练数据，并将其分成大小为32的batch，然后对于每个batch，执行一系列操作
        """
        super(SAE, self).__init__()
        self.dims = dims
        self.epochs_per_layer = epochs_per_layer
        self.lr = lr
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.num_layers = len(dims) - 1
        self.curr_layer = 1
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lossF = lossF
        for i in range(self.num_layers):
            self.encoders.append(nn.Linear(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
            self.decoders.append(nn.Linear(dims[i + 1], dims[i]))

        self.last_activation = nn.Tanh()
        self.activations = nn.ModuleList()
        self.activations_bk = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.activations.append(nn.LeakyReLU(0.05))
            self.activations_bk.append(nn.LeakyReLU(0.05))
        self.activations.append(nn.Tanh())
        self.activations_bk.append(nn.LeakyReLU(0.05))

    # def encoder_forward(self, x):
    #     z = x
    #     for i in range(self.curr_layer):
    #         z = self.activations[i](self.encoders[i](z))
    #     return z

    def forward(self, x):
        z = x
        ##
        if self.curr_layer == len(self.activations):
            for i in range(self.curr_layer-1):
                z = self.activations[i](self.bns[i](self.encoders[i](z)))
                # z = self.activations[i](self.encoders[i](z))
            z = self.encoders[self.curr_layer-1](z)
        else:
        ###
            for i in range(self.curr_layer):
                z = self.activations[i](self.bns[i](self.encoders[i](z)))
                # z = self.activations[i](self.encoders[i](z))
        for i in range(self.curr_layer-1):
            z = self.activations_bk[self.curr_layer - i - 1](self.decoders[self.curr_layer - i - 1](z))
        z = self.decoders[0](z)
        return z

    def train_forward(self, x):
        z = x
        c = torch.normal(0, torch.ones_like(x)*z.std()*0.05)
        z = z + c
        ##
        if self.curr_layer == len(self.activations):
            for i in range(self.curr_layer - 1):
                z = self.activations[i](self.bns[i](self.encoders[i](z)))
                # z = self.activations[i](self.encoders[i](z))
            z = self.encoders[self.curr_layer - 1](z)
        else:
            ###
            for i in range(self.curr_layer):
                z = self.activations[i](self.bns[i](self.encoders[i](z)))
                # z = self.activations[i](self.encoders[i](z))
        for i in range(self.curr_layer - 1):
            z = self.activations_bk[self.curr_layer - i - 1](self.decoders[self.curr_layer - i - 1](z))
        z = self.decoders[0](z)
        return z

    def control_freeze(self, n):
        for name_, param_ in self.named_parameters():
            name = name_.split('.')  # name example: encoders.0.weight  encoders.0.bias
            if name[0] == 'encoders':
                if int(name[1]) < n:
                    param_.requires_grad = False

    def control_unfreeze(self):
        for name_, param_ in self.named_parameters():
            param_.requires_grad = True

    def set_curr_layer(self, n):
        self.curr_layer = n
        self.control_freeze(n - 1)

    def fit(self, train_data):
        """
        train_data: annData
        """
        dataloader = DataLoader(Cellset(train_data), batch_size=self.batch_size, shuffle=True)
        if self.lossF == 'L1':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()  # 使用SGD和MSE
        for i in range(self.num_layers):
            # 即 训练好自编码器之后，就将编码器部分固定，用上一次编码器的output作下一次训练的input
            self.set_curr_layer(i + 1)
            optimizer = SGD(self.parameters(), lr=self.lr)
            for epoch in range(self.epochs_per_layer[i]):
                tt_loss = 0
                tt = 0
                for batch in dataloader:
                    x = batch[0]
                    optimizer.zero_grad()
                    x_hat = self.train_forward(x)  # 使用自编码器进行前向传播，得到重建的输出x_hat
                    loss = criterion(x_hat, x)  # 计算重建输出x_hat与原始输入x之间的MSE损失
                    loss.backward()  # 将loss作为反向传播的误差信号
                    optimizer.step()  # 反向传播，更新自编码器的参数
                    tt += 1
                    tt_loss += loss.item()
                print("layer:", i, "epoch:", epoch, "average_loss:", round(tt_loss / tt, 4))

    def save_para(self, name):
        torch.save(self.state_dict(), os.path.join(self.save_dir, name))

    def load_para(self, name):
        self.load_state_dict(torch.load(os.path.join(self.save_dir, name)))

if __name__ == '__main__':
    # raise KeyboardInterrupt
    dims = [64, 48, 36, 24, 12]
    epoches = [20,20,20,20,20]
    sae = SAE(dims, epoches, 0.05)
    # print(list(sae.parameters()))
    # sae.set_curr_layer(1)
    # print(list(sae.parameters()))
    sae(torch.tensor(range(64)).float())

    X = np.random.normal(0,1,(28,64)).astype(np.float32)
    obs = {'batch':np.random.randint(0,5, 28)}
    data = sc.AnnData(X, obs)
    # dataset = Cellset(data)
    # dataloader = DataLoader(dataset, 3, True)
    # for batch in dataloader:
    #     print(batch)
    #     break
    sae.fit(data)