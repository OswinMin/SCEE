import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Simdata(Dataset):
    def __init__(self, ann):
        self.x = torch.cat((ann[0], ann[1]))
        self.n = ann.shape[1]
        self.batch = torch.cat((torch.ones(self.n), -torch.ones(self.n))).reshape(-1,1)
        self.x = torch.cat((self.x, self.batch), axis=1)
        self.x = self.x.float()

    def __getitem__(self, item):
        x = self.x[item, :-1]
        y = self.x[item, -1]
        return x, y

    def __len__(self):
        return self.x.shape[0]

def gen(n, d):
    mu = torch.randn(d)
    A = torch.randn(d, d)
    sigma = torch.matmul(A, A.T)
    generator = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    simdata1 = generator.sample((1, n))
    mu = torch.randn(d) + 1
    A = torch.randn(d, d)
    sigma = torch.matmul(A, A.T)
    generator = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    simdata2 = generator.sample((1, n))
    return torch.cat([simdata1, simdata2])

if __name__ == '__main__':
    sim = gen(10,2)
    sm = Simdata(sim)
    data = DataLoader(sm, 2, shuffle=True)
    for x,y in data:
        print(x.shape, y.shape)
        break