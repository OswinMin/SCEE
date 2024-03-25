import numpy as np
import scanpy as sc
from network import *
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
import warnings
import umap
import seaborn as sns
import pandas as pd
import desc
warnings.filterwarnings('ignore')

def umap_draw(result):
    umap1 = umap.UMAP(n_components=2)
    X_umap = umap1.fit_transform(result)
    umap_data = pd.DataFrame(np.hstack([X_umap, batch_info.reshape(-1, 1)]), columns=['Dim1', 'Dim2', 'batch'])
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=umap_data, hue='batch', x='Dim1', y='Dim2')
    plt.title('Umap Visualization of features')
    plt.legend(loc='best')
    plt.savefig('Umap visualization of features.jpg')
    plt.show()

if __name__ == '__main__':
    # raise KeyboardInterrupt
    data_st = sc.read_h5ad("D:\\Mypy\\desc\\torch_version\\data\\brain_st_cortex.h5ad")
    data_sc = sc.read_h5ad("D:\\Mypy\\desc\\torch_version\\data\\brain_sc.h5ad")

    sc.pp.filter_cells(data_st, min_genes=200)
    sc.pp.filter_genes(data_st, min_cells=3)
    sc.pp.normalize_total(data_st, target_sum=1e4)
    sc.pp.log1p(data_st)
    sc.pp.highly_variable_genes(data_st, min_mean=0.0125, max_mean=3, min_disp=0.5)
    data_st = data_st[:, data_st.var.highly_variable]

    sc.pp.filter_cells(data_sc, min_genes=200)
    sc.pp.filter_genes(data_sc, min_cells=3)
    sc.pp.normalize_total(data_sc, target_sum=1e4)
    sc.pp.log1p(data_sc)
    sc.pp.highly_variable_genes(data_sc, min_mean=0.0125, max_mean=3, min_disp=0.5)
    data_sc = data_sc[:, data_sc.var.highly_variable]

    data_sc.obs_names_make_unique()
    data_st.obs_names_make_unique()
    data = sc.AnnData.concatenate(data_st, data_sc)

    clf = TruncatedSVD(500)
    Xpca = clf.fit_transform(data.X)

    dims = [Xpca.shape[1], 256, 128, 128]
    epoches = [300, 300, 300, 20, 20]
    batch_info = data.obs.batch.values.astype(int)

    desc1 = desc.train(data, dims=[data.shape[1],128,64], n_neighbors=18,
                      batch_size=32, louvain_resolution=[0.6], do_umap=True,
                      use_ae_weights=False, use_GPU=False)