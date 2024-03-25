import os
import matplotlib
import matplotlib.pyplot as plt
from time import time as get_time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from SAE import SAE
from Cellset import Cellset
from sklearn.cluster import KMeans
import pandas as pd
import scanpy as sc
from natsort import natsorted
from time import time


# try:
#     from .SAE import SAE # this is for installing package
# except:
# from SAE import SAE # this is for testing whether DescModel work or not

random.seed(202308)
np.random.seed(202308)
torch.manual_seed(202308)
def clearCache(file:str=None):
    if file is None:
        skip

class ClusteringLayer(nn.Module):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    def __init__(self, n_clusters, n_features, weights=None, alpha=1.0,
                 ):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.n_features = n_features

        # 生成 n_clusters * n_features 的权重作为聚类中心
        # self.clusters = nn.Parameter(torch.Tensor(self.n_clusters, n_features))
        if self.initial_weights is not None:
            self.clusters = nn.Parameter(torch.arctanh(torch.Tensor(self.initial_weights)))
            del self.initial_weights
        else:
            self.clusters = nn.Parameter(torch.Tensor(1, self.n_clusters, n_features))
            # 使用 xavier 方法初始化 clusters
            nn.init.xavier_uniform_(self.clusters)

    def forward(self, inputs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - torch.tanh(self.clusters)), dim=2) / self.alpha))
        # q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = (q.T / torch.sum(q, dim=1)).T
        return q

    def get_config(self):
        config = {'n_clusters':self.n_clusters, 'alpha':self.alpha,
                  'n_features':self.n_features}
        # base_config = super(ClusteringLayer, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        # shape=(n_samples, n_features)
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

# class ClusteringLayerGaussian(nn.Module):
#     def __init__(self):
#         super(ClusteringLayerGaussian, self).__init__()
#
#     def forward(self, inputs):
#         sigma = 1.
#         q = torch.sum(torch.exp_())


def KLDiv(x, y):
    kl = torch.mul(y, torch.log(y) - torch.log(x))
    kl = kl.sum()
    return kl


class DescModel(object):
    def __init__(self,
                 dims, # 预训练的SAE每一层encoder网络大小
                 x, # 数据矩阵，细胞数 * 基因数
                 batch, # 数据注释，每个细胞来自的 batch 类
                 alpha=1., # 超参 重构损失系数
                 alpha1=1., # 超参 q 次幂
                 alpha2=1., # 超参 软距离系数
                 name='demo',
                 tol=0.005, # 临界值
                 init='xavier', # 默认初始化参数方法
                 louvain_resolution=1.0,  # resolution for louvain
                 n_neighbors=10,  # louvain parameters
                 pretrain_epochs=300,  # epoch for autoencoder
                 batch_size=64,  # batch_size for autoencoder and model
                 save_length=10,
                 random_seed=20230420,
                 activation='relu',
                 actincenter="tanh",  # activation for the last layer in encoder, and first layer in the decoder
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 use_ae_weights=False,
                 use_fit1_saved=False,
                 sae_lossF='MSE',
                 save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp",
                 # kernel_clustering="t"
                 ):
        # 新建临时文件夹保存训练数据
        if not os.path.exists(save_dir):
            print("Create the directory:" + str(save_dir) + " to save result")
            os.mkdir(save_dir)

        self.name = name
        self.dims = dims
        self.num_layer = len(dims) - 1
        self.x = x
        self.alpha = alpha
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.tol = tol
        self.init = init
        self.save_length = save_length
        self.resolution = louvain_resolution
        self.n_neighbors = n_neighbors
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.activation = activation


        self.activations = nn.ModuleList()
        for i in range(self.num_layer - 1):
            self.activations.append(nn.ReLU())
        self.activations.append(nn.Tanh())

        self.actincenter = actincenter
        self.drop_rate_SAE = drop_rate_SAE
        self.is_stacked = is_stacked
        self.use_earlyStop = use_earlyStop
        self.use_ae_weights = use_ae_weights
        self.save_encoder_weights = save_encoder_weights
        self.save_encoder_step = save_encoder_step
        self.save_dir = save_dir
        self.sae_lossF = sae_lossF
        # self.kernel_clustering = kernel_clustering
        self.encoder = None # 编码器
        self.sae = None # 编码器+解码器
        self.model = None # 整个模型
        self.clustering_layer = None # 映射到概率
        self.ann = sc.AnnData(self.x) # x 和 batch 生成 anndata
        self.ann.obs['batch'] = batch # 添加 batch 信息
        self.dataset = Cellset(self.ann) # 用 x 生成 dataset
        self.dataloader = DataLoader(self.dataset, self.batch_size,
                                     shuffle=True, drop_last=False)
        # 用 dataset 生成 dataloader
        self.q = None
        self.use_fit1_saved = use_fit1_saved
        # self.batches = None
        self.pretrain()

    def pretrain(self):
        sae = SAE(dims=self.dims,
                  epochs_per_layer=self.pretrain_epochs,
                  batch_size=self.batch_size,
                  lr=0.05,
                  save_dir=self.save_dir,
                  lossF=self.sae_lossF
                  )
        t0 = time()

        # 训练 stacked auto encoder
        # 使用已经保存的网络
        if not self.use_fit1_saved:
            if self.use_ae_weights:
                print(f"Checking whether {str(os.path.join(self.save_dir, 'ae_weights.h5'))} exists in the directory")
                if not os.path.isfile(self.save_dir + f"/ae_weights_{self.name}.h5"):
                    # 保存的文件不存在，则重新训练
                    print(f"The file ae_weights_{self.name}.h5 is not exits")
                    sae.fit(self.ann)
                    self.sae = sae
                    self.encoder = sae.encoders
                    self.activations = sae.activations
                else:
                    sae.load_para(f"ae_weights_{self.name}.h5")
                    self.sae = sae
                    self.encoder = sae.encoders
                    self.activations = sae.activations
            else:
                print("use_ae_weights=False, the program will rerun autoencoder")
                sae.fit(self.ann)
                self.sae = sae
                self.encoder = sae.encoders
                self.activations = sae.activations
        else:
            sae.load_state_dict(torch.load(os.path.join("fit1tmp", f"sae_{self.name}.h5")))
            self.sae = sae
            self.encoder = sae.encoders
            self.activations = sae.activations

        print('Pretraining time is', get_time() - t0)
        # 训练好的 encoder 保存起来
        if not os.path.isfile(os.path.join(self.save_dir, "ae_weights.h5")):
            self.sae.save_para(f'ae_weights_{self.name}.h5')
            # self.encoder.save_weights(os.path.join(self.save_dir, 'encoder_weights.h5'))
            print('Pretrained weights are saved to %s /ae_weights.h5' % self.save_dir)
        # self.autoencoder.save(os.path.join(self.save_dir, "autoencoder_model.h5"))
        # 通过 louvain 确定聚类个数

        self.sae.curr_layer = self.sae.num_layers
        features = self.extract_features(torch.tensor(self.x).float()) # 获取隐藏层（降维后）的数据矩阵
        features = features.detach().numpy()
        print("...number of clusters is unknown, Initialize cluster centroid using louvain method")
        adata0 = sc.AnnData(features)
        # 此时 adata0.X 是数据矩阵，obs 是不同的细胞，var 是隐藏层数据
        # 如果细胞个数太多，那就拿 200000 个出来找聚类中心（不然算太久）
        if adata0.shape[0] > 200000:
            np.random.seed(adata0.shape[0])  # set  seed
            adata0 = adata0[np.random.choice(adata0.shape[0], 200000, replace=False)]
        sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors, use_rep="X")
        # adata0.uns['neighbors'].get('*') 中保存两个数据矩阵 connectivities 和 distances
        # 用于下一步计算 louvain 得到的组别分类在 obs 的 louvain 中
        sc.tl.louvain(adata0, resolution=self.resolution)
        # 每个 observation 的聚类分类
        Y_pred_init = adata0.obs['louvain']
        self.init_pred = np.asarray(Y_pred_init, dtype=int)
        # 如果只有一个聚类那没信息，要求提高 resolution 重新来
        if np.unique(self.init_pred).shape[0] <= 1:
            exit("Error: There is only a cluster detected. The resolution:" + str(
                self.resolution) + "is too small, please choose a larger resolution!!")
        # 根据 init_pred（louvain结果）计算每个分组的中心
        features = pd.DataFrame(adata0.X, index=np.arange(0, adata0.shape[0]))
        Group = pd.Series(self.init_pred, index=np.arange(0, adata0.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        # 聚类个数
        self.n_clusters = cluster_centers.shape[0]
        # 聚类初始化中心，用来初始化 clusterlayer
        self.init_centroid = np.array([cluster_centers])
        self.init_centroid_copy = self.init_centroid.copy()
        # 聚类层，隐藏层 z -> p
        # n_samples x n_features -> n_samples x n_clusters
        if self.use_fit1_saved:
            xx = torch.load(os.path.join("fit1tmp", f"clus_{self.name}.h5"))
            self.clustering_layer = ClusteringLayer(xx['clusters'].shape[1], features.shape[1],
                                                    weights=None, alpha=self.alpha1)
            self.clustering_layer.load_state_dict(xx)
        else:
            self.clustering_layer = ClusteringLayer(self.n_clusters, features.shape[1],
                                               weights=self.init_centroid, alpha=self.alpha1)
        # 组装 encoder 和 clustering_layer
        self.model = nn.Sequential(
            self.encoder,
            self.sae.decoders,
            self.clustering_layer,
        )

        default_q = torch.ones((self.batch_size, self.n_clusters), dtype=torch.float32)/self.n_clusters
        default_batches = torch.zeros((self.batch_size, self.dataset.batch_num), dtype=torch.float32)
        # default_batches[:, 0] += 1
        self.q = []
        self.batches = []
        for i in range(self.save_length):
            self.q.append(default_q)
            self.batches.append(default_batches)

    # 此处注意和 sae 对应接口
    # def extract_features(self, x):
    #     return self.sae.encoder_forward(x)

    def extract_features(self, x):
        return self.encoder_forward(x)

    def encoder_forward(self, x):
        z = x
        for i in range(self.num_layer):
            z = self.activations[i](self.sae.bns[i](self.encoder[i](z)))
        # z = self.encoder[self.num_layer-1](z)
        return z

    def model_forward(self, x):
        z = self.encoder_forward(x)
        z = self.clustering_layer(z)
        return z

    # 得到到每个细胞的分类分布
    def model_predict(self, x):
        q = self.model_forward(x)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def update_data(self, q, batches):
        self.q.pop(0)
        self.batches.pop(0)
        self.q.append(q.detach())
        self.batches.append(batches.detach())

    def cal_EO(self):
        batches = torch.concat(self.batches)
        q = torch.concat(self.q)
        qc = q.sum(0)
        qc = qc / qc.sum()
        qb = batches.sum(0)
        qb = qb / qb.sum()
        E = torch.mul(qc.reshape((-1,1)), qb)
        O = torch.matmul(q.T, batches)
        O = O / O.sum()
        return E, O

    def cal_entro(self, q):
        return (q * torch.log(q)).sum(1).mean()

    def fit(self, maxiter=1e3, save_encoder_step=100, lr=0.05, update_interval=1000,
            use_saved=False):
        if use_saved:
            return
        save_dir = self.save_dir
        # init 时 pretrain 已经对 encoder+clustering_layer 进行了预训练和设置
        # y_pred_last = np.copy(self.init_pred)
        # index_array = np.arange(self.x.shape[0])
        epoches = maxiter // (self.x.shape[0] // self.batch_size + 1) + 1
        ite = 0
        self.sae.control_unfreeze()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # desc loss function KL(p||q)
        # 后续需要加入 KL(O||E)
        # batch training 时计算一部分样本，考虑计算整体
        loss_fun = nn.KLDivLoss(reduction='sum')
        sae_criterion = nn.MSELoss()
        ##### ATTENTION !!!!!
        ##### KLDivLoss only receive distribution in log space, must use torch.log(p) rather than p
        for ep in range(int(epoches)):
            # 每轮循环一遍 cellset 中所有数据
            for xs, batches in self.dataloader:
                optimizer.zero_grad()
                ite += 1
                # 隔一段时间存一下 encoder
                if self.save_encoder_weights and \
                    ite % (save_encoder_step) == 0:
                    self.encoder.save_weights(
                        os.path.join(save_dir,
                                     f'encoder_resolution_{self.resolution}_{ite}.h5')
                    )
                    print(f'Fine tuning encoder weights are saved to '
                          f'encoder_resolution_{self.resolution}_{ite}.h5')
                # 计算 q，p 和损失函数并训练网络
                q = self.model_forward(xs)
                p = self.target_distribution(q)
                self.update_data(q, batches)                                          # 更新输出
                E, O = self.cal_EO()                                                    # 计算 E, O

                z = self.encoder_forward(xs)
                xshat = self.sae.train_forward(xs)
                # batches__ = torch.concat(self.batches)
                # q__ = torch.concat(self.q)
                # qc__ = q__.sum(0)
                # qc__ = qc__ / qc__.sum()
                # qb__ = batches__.sum(0)
                # qb__ = qb__ / qb__.sum()
                # E = torch.mul(qc__.reshape((-1, 1)), qb__)
                # O = torch.matmul(q__.T, batches__)
                # O = O / O.sum()

                # loss1 = KLDiv(q, p)
                loss2 = KLDiv(O, E)
                loss3 = torch.mean(torch.sum(torch.square(torch.unsqueeze(z, dim=1) -
                                    torch.tanh(self.clustering_layer.clusters)), dim=2) * q)
                loss4 = sae_criterion(xshat, xs)
                loss5 = self.cal_entro(q)
                # loss = loss1 + loss2 + self.alpha2 * torch.pow(loss3, 0.5) + self.alpha * loss4
                loss = 40 * loss2 + self.alpha2 * loss3 + \
                       self.alpha * loss4 + loss5 * 0.05
                if ite % 767 == 0:
                    print(40 * loss2, self.alpha2 * loss3, self.alpha * loss4, loss5 * 0.05)
                # loss = loss1 + loss2 + self.alpha * loss4
                loss.backward()
                optimizer.step()
                if ite % update_interval == 0:
                    print(f"Current iter average KL loss {round(loss.item(),4)}")
                if ite > maxiter:
                    break
            if ite > maxiter:
                break
        # 最后保存一遍 encoder
        # self.encoder.save(os.path.join(self.save_dir, "encoder_model.h5"))
        torch.save(self.sae.state_dict(), os.path.join("fit1tmp", f"sae_{self.name}.h5"))
        torch.save(self.clustering_layer.state_dict(), os.path.join("fit1tmp", f"clus_{self.name}.h5"))

        # 提取最终的隐藏层和到每类的分类 q
        Embedded_z = self.extract_features(torch.tensor(self.x).float())
        q = self.model_predict(torch.tensor(self.x).float())
        return Embedded_z, q

    def forward1(self):
        x = self.x
        batch = self.ann.obs.batch.values.astype(int)
        # batch = np.eye(len(np.unique(batch)))[batch]
        z = self.extract_features(torch.tensor(x).float()).detach().numpy()
        q = self.model_forward(torch.tensor(x).float()).detach().numpy()
        p = self.target_distribution(q)
        cl = torch.tanh(self.clustering_layer.clusters).detach().numpy()[0,:,:]
        cl = ((p.T @ z).T / p.sum(0)).T
        c = p @ cl
        ls = []
        ls.append(((p[batch == 0., :].T @ c[batch == 0., :]).T / p[batch == 0., :].sum(0)).T)
        ls.append(((p[batch == 1., :].T @ c[batch == 1., :]).T / p[batch == 1., :].sum(0)).T)
        c[batch==0., :] = c[batch==0., :] - p[batch==0., :] @ ls[0]
        c[batch==1., :] = c[batch==1., :] - p[batch==1., :] @ ls[1]
        return c

    def fit_second(self, maxiter=1e3, save_encoder_step=100, lr=0.0005, update_interval=100):
        save_dir = self.save_dir
        epoches = maxiter // (self.x.shape[0] // self.batch_size + 1) + 1
        ite = 0
        self.sae.control_unfreeze()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        for ep in range(int(epoches)):
            # 每轮循环一遍 cellset 中所有数据
            for xs, batches in self.dataloader:
                optimizer.zero_grad()
                ite += 1

                # 计算 q，p 和损失函数并微调网络
                q = self.model_forward(xs)
                p = self.target_distribution(q)

                loss = KLDiv(q, p)
                loss.backward()
                optimizer.step()
                if ite % update_interval == 0:
                    print(f"Current iter average KL loss {round(loss.item(),5)}")
                if ite > maxiter:
                    break
            if ite > maxiter:
                break

class PureDesc(object):
    def __init__(self,
                 dims, # 预训练的SAE每一层encoder网络大小
                 x, # 数据矩阵，细胞数 * 基因数
                 batch, # 数据注释，每个细胞来自的 batch 类
                 alpha=1., # 超参 q 次幂
                 name='demo',
                 tol=0.01, # 临界值
                 init='xavier', # 默认初始化参数方法
                 louvain_resolution=1.0,  # resolution for louvain
                 n_neighbors=10,  # louvain parameters
                 pretrain_epochs=300,  # epoch for autoencoder
                 batch_size=64,  # batch_size for autoencoder and model
                 save_length=10,
                 random_seed=20230420,
                 activation='relu',
                 actincenter="tanh",  # activation for the last layer in encoder, and first layer in the decoder
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 sae_lossF='MSE',
                 save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="fit1tmp"
                 ):
        self.name = name
        self.dims = dims
        self.num_layer = 1
        self.x = x
        self.alpha = alpha
        self.tol = tol
        self.init = init
        self.save_length = save_length
        self.resolution = louvain_resolution
        self.n_neighbors = n_neighbors
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.activation = activation

        self.activations = nn.ModuleList()
        for i in range(self.num_layer - 1):
            self.activations.append(nn.ReLU())
        self.activations.append(nn.Tanh())

        self.actincenter = actincenter
        self.drop_rate_SAE = drop_rate_SAE
        self.is_stacked = is_stacked
        self.use_earlyStop = use_earlyStop
        self.save_encoder_weights = save_encoder_weights
        self.save_encoder_step = save_encoder_step
        self.save_dir = save_dir
        self.sae_lossF = sae_lossF
        self.encoder = None # 编码器
        self.sae = None # 编码器+解码器
        self.model = None # 整个模型
        self.ann = sc.AnnData(self.x) # x 和 batch 生成 anndata
        self.ann.obs['batch'] = batch # 添加 batch 信息
        self.dataset = Cellset(self.ann) # 用 x 生成 dataset
        self.dataloader = DataLoader(self.dataset, self.batch_size,
                                     shuffle=True, drop_last=False)
        # 用 dataset 生成 dataloader
        self.q = None
        self.pretrain()

    def pretrain(self):
        sae = SAE(dims=[self.dims[-1], self.dims[-1]],
                  epochs_per_layer=[1, 1],
                  batch_size=self.batch_size,
                  lr=0.05,
                  save_dir=self.save_dir,
                  lossF=self.sae_lossF
                  )
        # 训练 stacked auto encoder
        # 使用已经保存的网络
        # sae.load_state_dict(torch.load(os.path.join("fit1tmp", f"sae_{self.name}.h5")))
        self.sae = sae
        self.encoder = sae.encoders
        self.activations = sae.activations
        self.sae.control_freeze(len(self.dims) + 1)
        self.sae.encoders[0].bias = torch.nn.Parameter(torch.zeros_like(self.sae.encoders[0].bias))
        self.sae.decoders[0].bias = torch.nn.Parameter(torch.zeros_like(self.sae.decoders[0].bias))
        self.sae.encoders[0].weight = torch.nn.Parameter(torch.eye(self.sae.encoders[0].weight.shape[0]))
        self.sae.decoders[0].weight = torch.nn.Parameter(torch.eye(self.sae.decoders[0].weight.shape[0]))

        # 通过 louvain 确定聚类个数
        # self.sae1.curr_layer = self.sae1.num_layers
        features = self.extract_features(torch.tensor(self.x).float()) # 获取隐藏层（降维后）的数据矩阵
        features = features.detach().numpy()
        adata0 = sc.AnnData(features)
        # 此时 adata0.X 是数据矩阵，obs 是不同的细胞，var 是隐藏层数据
        # 如果细胞个数太多，那就拿 200000 个出来找聚类中心（不然算太久）
        if adata0.shape[0] > 200000:
            np.random.seed(adata0.shape[0])  # set  seed
            adata0 = adata0[np.random.choice(adata0.shape[0], 200000, replace=False)]
        sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors, use_rep="X")
        # adata0.uns['neighbors'].get('*') 中保存两个数据矩阵 connectivities 和 distances
        # 用于下一步计算 louvain 得到的组别分类在 obs 的 louvain 中
        sc.tl.louvain(adata0, resolution=self.resolution)
        # 每个 observation 的聚类分类
        Y_pred_init = adata0.obs['louvain']
        self.init_pred = np.asarray(Y_pred_init, dtype=int)
        # 如果只有一个聚类那没信息，要求提高 resolution 重新来
        if np.unique(self.init_pred).shape[0] <= 1:
            exit("Error: There is only a cluster detected. The resolution:" + str(
                self.resolution) + "is too small, please choose a larger resolution!!")
        # 根据 init_pred（louvain结果）计算每个分组的中心
        features = pd.DataFrame(adata0.X, index=np.arange(0, adata0.shape[0]))
        Group = pd.Series(self.init_pred, index=np.arange(0, adata0.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        # 聚类个数
        self.n_clusters = cluster_centers.shape[0]
        # 聚类初始化中心，用来初始化 clusterlayer
        self.init_centroid = np.array([cluster_centers])
        self.init_centroid_copy = self.init_centroid.copy()
        # 聚类层，隐藏层 z -> p
        # n_samples x n_features -> n_samples x n_clusters
        self.clustering_layer = ClusteringLayer(self.n_clusters, features.shape[1],
                                           weights=self.init_centroid, alpha=self.alpha)
        # 组装 encoder 和 clustering_layer
        self.model = nn.Sequential(
            self.encoder,
            self.sae.decoders,
            self.clustering_layer,
        )

        # default_q = torch.ones((self.batch_size, self.n_clusters), dtype=torch.float32)/self.n_clusters
        # default_batches = torch.zeros((self.batch_size, self.dataset.batch_num), dtype=torch.float32)
        # # default_batches[:, 0] += 1
        # self.q = []
        # self.batches = []
        # for i in range(self.save_length):
        #     self.q.append(default_q)
        #     self.batches.append(default_batches)

    # 此处注意和 sae 对应接口
    # def extract_features(self, x):
    #     return self.sae.encoder_forward(x)

    def extract_features(self, x):
        return self.encoder_forward(x)

    def encoder_forward(self, x):
        z = x
        # for i in range(self.num_layer):
        z = self.encoder[0](z)
        # z = self.encoder[self.num_layer-1](z)
        return z

    def model_forward(self, x):
        z = self.encoder_forward(x)
        z = self.clustering_layer(z)
        return z

    # 得到到每个细胞的分类分布
    def model_predict(self, x):
        q = self.model_forward(x)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def update_data(self, q, batches):
        self.q.pop(0)
        self.batches.pop(0)
        self.q.append(q.detach())
        self.batches.append(batches.detach())

    def cal_entro(self, q):
        return (q * torch.log(q)).sum(1).mean()

    def fit(self, maxiter=1e3, save_encoder_step=100, lr=0.05, update_interval=1000,
            use_saved=False):
        save_dir = self.save_dir
        epoches = maxiter // (self.x.shape[0] // self.batch_size + 1) + 1
        ite = 0
        self.sae.control_unfreeze()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        loss_fun = nn.KLDivLoss(reduction='sum')
        sae_criterion = nn.MSELoss()
        ##### ATTENTION !!!!!
        ##### KLDivLoss only receive distribution in log space, must use torch.log(p) rather than p
        for ep in range(int(epoches)):
            # 每轮循环一遍 cellset 中所有数据
            for xs, batches in self.dataloader:
                optimizer.zero_grad()
                ite += 1

                # 计算 q，p 和损失函数并训练网络
                q = self.model_forward(xs)
                p = self.target_distribution(q)
                # xshat = self.sae.train_forward(xs)
                # loss1 = sae_criterion(xshat, xs)

                loss = KLDiv(q, p)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if ite <= 1:
                        ty = self.model_predict(torch.tensor(self.x)).detach().numpy()
                    ty_ = self.model_predict(torch.tensor(self.x)).detach().numpy()
                    if ite > 1:
                        if (ty != ty_).sum() / len(ty) < self.tol:
                            ite = maxiter + 5
                        ty = ty_

                if ite % update_interval == 0:
                    print(f"Current iter average KL loss {round(loss.item(),4)}")
                if ite > maxiter:
                    break
            if ite > maxiter:
                break
        # 提取最终的隐藏层和到每类的分类 q
        Embedded_z = self.extract_features(torch.tensor(self.x).float())
        q = self.model_predict(torch.tensor(self.x).float())
        return Embedded_z, q

if __name__ == '__main__':
    # raise KeyboardInterrupt
    X = np.random.normal(0, 1, (28, 64)).astype(np.float32)
    obs = np.random.randint(0,5, 28)
    dims = [64, 48, 36, 24, 12]
    epoches = [20, 20, 20, 20, 20]
    desc = DescModel(dims=dims, x=X, louvain_resolution=0.8, use_ae_weights=True, batch=obs, pretrain_epochs=epoches)
    t0 = get_time()
    Embedded_z, q_pred = desc.fit(maxiter=30)
    print('clustering time: ', (get_time() - t0))