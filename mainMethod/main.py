import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

from network import *
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
import warnings
import umap
import seaborn as sns
import pandas as pd
import os
warnings.filterwarnings('ignore')

def umap_draw(result, Xpca, title='Umap Visualization of features',
              save=False, save_dir='Umap visualization of features.jpg',
              ):
    # clf = TruncatedSVD(500)
    # Xpca = clf.fit_transform(result.X)
    umap1 = umap.UMAP(n_components=2)
    X_umap = umap1.fit_transform(Xpca)
    plt.figure(figsize=(6, 6))
    if 'batch' in result.obs:
        umap_data = pd.DataFrame(np.hstack([X_umap,
                        result.obs.batch.values.astype(int).reshape(-1, 1)]),
                        columns=['Dim1', 'Dim2', 'batch'])
        sns.scatterplot(data=umap_data, hue='batch', x='Dim1', y='Dim2',
                        s=2)
    else:
        umap_data = pd.DataFrame(np.hstack([X_umap]), columns=['Dim1', 'Dim2'])
        sns.scatterplot(data=umap_data, x='Dim1', y='Dim2',
                        s=2)
    plt.title(title)
    plt.legend(loc='best')
    if save:
        plt.savefig(save_dir)
    plt.show()

data_st = sc.read_h5ad("D:\\Mypy\\desc\\torch_version\\data\\brain_st_cortex.h5ad")
data_sc = sc.read_h5ad("D:\\Mypy\\desc\\torch_version\\data\\brain_sc.h5ad")
data = sc.AnnData.concatenate(data_st, data_sc)

data_sc = data_sc[:, data.var_names]
data_st = data_st[:, data.var_names]

### spatial preprocessing
sc.pp.filter_cells(data_st, min_genes=200)
# sc.pp.filter_genes(data_st, min_cells=3)
sc.pp.normalize_total(data_st, target_sum=1e4)
sc.pp.log1p(data_st)
sc.pp.highly_variable_genes(data_st, #min_mean=0.0125, max_mean=3, min_disp=0.5,
                            n_top_genes=500)
sc.pp.scale(data_st)
# data_st1 = data_st[:, data_st.var.highly_variable]

### single cell preprocessing
sc.pp.filter_cells(data_sc, min_genes=200)
# sc.pp.filter_genes(data_sc, min_cells=3)
sc.pp.normalize_total(data_sc, target_sum=1e4)
sc.pp.log1p(data_sc)
sc.pp.highly_variable_genes(data_sc, #min_mean=0.0125, max_mean=3, min_disp=0.5,
                            n_top_genes=500)
sc.pp.scale(data_sc)
raise KeyboardInterrupt
s1 = data_sc.var.highly_variable[data_sc.var.highly_variable]
s2 = data_st.var.highly_variable[data_st.var.highly_variable]
# data = sc.AnnData.concatenate(data_st, data_sc)[:, np.unique(pd.concat([s1,s2]).index)]
data = sc.AnnData.concatenate(data_sc, data_st)[:, np.unique(pd.concat([s1,s2]).index)]

# raise KeyboardInterrupt
# umap_draw(data, data.X, title='origin data after concatenating')

####
# data_sc1 = data_sc[:, data_sc.var.highly_variable]
data_st1 = data_st[:, np.unique(pd.concat([s1,s2]).index)]
# umap_draw(data_st1, data_st1.X, title='origin spatial data')
# umap_draw(data_sc1, data_sc1.X, title='origin single cell data')
####

X = data.X.copy()

dims = [X.shape[1], 256, 64]
epoches = [400, 400]
batch_info = data.obs.batch.values.astype(int)

desc = DescModel(x=X, dims=dims, louvain_resolution=0.6, use_ae_weights=True,
                 pretrain_epochs=epoches, batch=batch_info, name='twostage',
                 alpha=0.5, alpha1=1, alpha2=0.2, sae_lossF='MSE', save_length=5,
                 use_fit1_saved=False)
# desc.fit(maxiter=10000, update_interval=50, lr=0.001)
result = desc.extract_features(torch.tensor(X)).detach().numpy()
# umap_draw(data, result, title='data after sae')
desc.fit(maxiter=4000, update_interval=50, lr=0.0001)
result = desc.extract_features(torch.tensor(X)).detach().numpy()
umap_draw(data, result, title='data after fit1 extract features')
result = desc.forward1()
umap_draw(data, result, title='data after fit1 forward1')
raise KeyboardInterrupt

X1 = desc.forward1(torch.tensor(X)).detach().numpy()
umap_draw(data, X1, title='data after fit1')
pdesc = PureDesc(x=X1, dims=dims, louvain_resolution=0.6,
                 pretrain_epochs=epoches, batch=batch_info, name='twostage',
                 alpha=1., sae_lossF='MSE', save_length=3, tol=0.005)
raise KeyboardInterrupt

pdesc.fit(maxiter=100, update_interval=2, lr=0.0005, )
result = pdesc.extract_features(torch.tensor(X1)).detach().numpy()
umap_draw(data, result, title='data after fit2')

raise KeyboardInterrupt

desc.fit(maxiter=5000, update_interval=100, lr=0.001)

def test(alpha, alpha1, alpha2, folder, name='demo1'):
    desc = DescModel(x=X, dims=dims, louvain_resolution=0.6, use_ae_weights=True,
                     pretrain_epochs=epoches, batch=batch_info, name=name,
                     alpha=alpha, alpha1=alpha1, alpha2=alpha2)
    for i in range(20):
        desc.fit(maxiter=500, update_interval=100)
        result = desc.extract_features(torch.tensor(X)).detach().numpy()
        umap_draw(data, result, title=f'{alpha}-{alpha1}-{alpha2} {i + 1}*500 iter', save=True,
                  save_dir=os.path.join(folder, f'{i + 1}-500 iter {alpha}-{alpha1}-{alpha2}-{name}.jpg'))

alpha1 = 2.
for alpha in [3., 5., 10.]:
    for alpha2 in [0.3, 0.5, 1]:
        for name in ['demo2', 'demo3']:
            save_dir = os.path.join('fig', f"{alpha}-{alpha1}-{alpha2}-{name}")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            test(alpha, alpha1, alpha2, save_dir, name)

desc = DescModel(x=X, dims=dims, louvain_resolution=0.6, use_ae_weights=True,
                 pretrain_epochs=epoches, batch=batch_info, name='demo2',
                 alpha=4., alpha1=2., alpha2=0.5)

# desc = DescModel(x=X, dims=dims, louvain_resolution=0.6, use_ae_weights=False,
#                  pretrain_epochs=epoches, batch=batch_info, name='demo2',
#                  alpha=4., alpha1=2., alpha2=0.5)
result = desc.extract_features(torch.tensor(X)).detach().numpy()
umap_draw(data, result, title='data after sae')



# raise KeyboardInterrupt
# for i in range(25):
#     desc.fit(maxiter=500, update_interval=100)
#     result = desc.extract_features(torch.tensor(X)).detach().numpy()
#     umap_draw(data, result, title=f'train {i+1}*500 iter', save=True,
#               save_dir=f'fig\\umap_iter_{i+1}.jpg')
# desc.fit(maxiter=1e3, update_interval=200)

# raise KeyboardInterrupt
# result1 = desc.model_forward(torch.tensor(Xpca)).detach().numpy()
# result = desc.extract_features(torch.tensor(X)).detach().numpy()
# umap_draw(data, result, title='data after train 10e3')
# 检查聚类中心
cl = desc.clustering_layer.clusters.detach().numpy()[0,:,:]
X_umap = umap1.fit_transform(cl)
plt.figure(figsize=(6, 6))
umap_data = pd.DataFrame(np.hstack([X_umap]), columns=['Dim1', 'Dim2'])
sns.scatterplot(data=umap_data, x='Dim1', y='Dim2',
                s=50)
plt.show()

import numpy as np
for i in range(10):
    pdesc.fit(maxiter=20, update_interval=2, lr=0.0001)
    result = pdesc.extract_features(torch.tensor(X)).detach().numpy()
    umap1 = umap.UMAP(n_components=2)
    X_umap = umap1.fit_transform(np.vstack([result, cl]))
    plt.figure(figsize=(6, 6))
    umap_data = pd.DataFrame(np.hstack([X_umap[:result.shape[0],:],
                                        batch_info.reshape(-1,1)]),
                             columns=['Dim1', 'Dim2', 'batch'])
    sns.scatterplot(data=umap_data, x='Dim1', y='Dim2', s=2, hue='batch')
    umap_data_cl = pd.DataFrame(X_umap[result.shape[0]:,:],
                             columns=['Dim1', 'Dim2'])
    sns.scatterplot(data=umap_data_cl, x='Dim1', y='Dim2', s=50, c=[0.5]*17)
    plt.title(f'{i}')
    plt.show()


result = pdesc.extract_features(torch.tensor(X)).detach().numpy()
ty = pdesc.model_predict(torch.tensor(X)).detach().numpy()
umap1 = umap.UMAP(n_components=2)
X_umap = umap1.fit_transform(np.vstack([result]))
plt.figure(figsize=(6, 6))
umap_data = pd.DataFrame(np.hstack([X_umap,
                                    ty.reshape(-1,1)]),
                         columns=['Dim1', 'Dim2', 'batch'])
sns.scatterplot(data=umap_data, x='Dim1', y='Dim2', s=2, hue='batch',
                palette=sns.color_palette("hls", 16))
plt.show()

plt.figure(figsize=(6, 6))
umap_data = pd.DataFrame(np.hstack([X_umap,
                                    batch_info.reshape(-1,1)]),
                         columns=['Dim1', 'Dim2', 'batch'])
sns.scatterplot(data=umap_data, x='Dim1', y='Dim2', s=2, hue='batch',
                palette=sns.color_palette("hls", 2))
plt.show()

xx = pd.DataFrame(np.hstack([batch_info.reshape(-1,1), ty.reshape(-1,1)]), columns=['b','t'])