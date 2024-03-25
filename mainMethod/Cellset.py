import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class Cellset(Dataset):
    """
        Dataset for cell matrix x. Store x which is n_samples x n_genes.
        Arguments:
            ann: anndata, x stands for matrix, ann.obs['batch'] stands for different batch
        Return:
            A dataset class to further generate dataloader.
    """

    def __init__(self, ann):
        self.x = ann.X
        self.cellbatch = ann.obs['batch'].values.astype(int)
        self.batch_num = len(np.unique(self.cellbatch))

    def __getitem__(self, item):
        x = torch.zeros(self.batch_num, dtype=torch.float32)
        x[self.cellbatch[item]] += 1.
        return torch.tensor(self.x[item, :]).float(), x

    def __len__(self):
        return self.x.shape[0]