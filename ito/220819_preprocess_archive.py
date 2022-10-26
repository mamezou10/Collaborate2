import os
from this import s
# os.chdir('../')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import scanpy as sc
import torch
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import matplotlib.pyplot as plt
import numpy as np
import gdown

dir = "/mnt/Donald/ito/220819"
os.makedirs(dir)
os.chdir(dir)

sc.settings.set_figure_params(dpi=200, frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

# url = 'https://drive.google.com/uc?id=1ehxgfHTsMZXy6YzlFKGJOsBKQ5rrvMnd'
# output = 'pancreas.h5ad'
# gdown.download(url, output, quiet=False)

adata_all = sc.read('/mnt/Daisy/sc_BRAIN_GSE153424/count_adata.h5ad')
adata_all = adata_all[adata_all.obs.condition=="control"]
adata1 = remove_sparsity(adata_all)


adata_all = sc.read_h5ad("/mnt/Daisy/sc_BRAIN_GSE128855/adata.h5ad")
adata_all.X = adata_all.layers["count"]
adata2 = remove_sparsity(adata_all)

adata3 = sc.read_h5ad("/mnt/Donald/ito/220816/preprocessed_adata.h5ad")
adata3.X = adata3.layers["count"]
adata3 = remove_sparsity(adata3)

adata = adata1.concatenate([adata2, adata3], 
                            uns_merge="same", index_unique=None, batch_key="dataset", 
                            batch_categories=["GSE153424", "GSE128855", "query"])

bdata = adata.copy()
sc.pp.normalize_total(bdata)
sc.pp.log1p(bdata)
sc.pp.highly_variable_genes(bdata )

adata.var["highly_variable"] = bdata.var["highly_variable"]
adata = adata[:, adata.var["highly_variable"]]
# raw_adata = adata.copy()

df = pd.DataFrame(adata.obs[["CellTypes_Level1", "cluster", "dataset"]])
df.loc[df.dataset=="GSE153424", ["cell_type"]] = df.CellTypes_Level1[df.dataset=="GSE153424"]
df.loc[df.dataset=="GSE128855", ["cell_type"]] = df.cluster[df.dataset=="GSE128855"]

adata.obs["cell_type"] = list(df.cell_type)

source_adata = adata[~adata.obs[condition_key].isin(target_conditions)].copy()
target_adata = adata[adata.obs[condition_key].isin(target_conditions)].copy()


# ///////////////





sca.models.SCVI.setup_anndata(source_adata, batch_key=condition_key)

vae = sca.models.SCVI(
    source_adata,
    n_layers=2,
    encode_covariates=True,
    deeply_inject_covariates=False,
    use_layer_norm="both",
    use_batch_norm="none",
)
vae.train()


df = pd.DataFrame(source_adata.obs[["CellTypes_Level1", "cluster", "dataset"]])
df.loc[df.dataset=="GSE153424", ["cell_type"]] = df.CellTypes_Level1[df.dataset=="GSE153424"]
df.loc[df.dataset=="GSE128855", ["cell_type"]] = df.cluster[df.dataset=="GSE128855"]

source_adata.obs["cell_type"] = df.cell_type

reference_latent = sc.AnnData(vae.get_latent_representation())
reference_latent.obs["cell_type"] = source_adata.obs[cell_type_key].tolist()
reference_latent.obs["batch"] = source_adata.obs[condition_key].tolist()

sc.pp.neighbors(reference_latent, n_neighbors=8)
sc.tl.leiden(reference_latent)
sc.tl.umap(reference_latent)
sc.pl.umap(reference_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           )

ref_path = 'ref_model1/'
vae.save(ref_path, overwrite=True)
model = sca.models.SCVI.load_query_data(
    target_adata,
    ref_path,
    freeze_dropout = True,
)
model.train(max_epochs=200, plan_kwargs=dict(weight_decay=0.0))


# # adata1.obs[['CellTypes_Level1', 'CellTypes_Level3', 'CellTypes_Level4', 'CellTypes_Level4_zeisel','CellTypes_Level4_marty']]   



query_latent = sc.AnnData(model.get_latent_representation())
query_latent.obs['cell_type'] = target_adata.obs[cell_type_key].tolist()
query_latent.obs['batch'] = target_adata.obs[condition_key].tolist()
sc.pp.neighbors(query_latent)
sc.tl.leiden(query_latent)
sc.tl.umap(query_latent)
plt.figure()
sc.pl.umap(
    query_latent,
    color=["batch"],
    frameon=False,
    wspace=0.6,
)

surgery_path = 'surgery_model'
model.save(surgery_path, overwrite=True)

adata_full = source_adata.concatenate(target_adata)
full_latent = sc.AnnData(model.get_latent_representation(adata=adata_full))
full_latent.obs['cell_type'] = adata_full.obs[cell_type_key].tolist()
full_latent.obs['batch'] = adata_full.obs[condition_key].tolist()
sc.pp.neighbors(full_latent)
sc.tl.leiden(full_latent)
sc.tl.umap(full_latent)
plt.figure()
sc.pl.umap(
    full_latent,
    color=["batch", "cell_type"],
    frameon=False,
    wspace=0.6,
)





# /////////////////////



condition_key = 'dataset'
cell_type_key = 'cell_type'
target_conditions = ['query']
source_conditions = ["GSE153424", "GSE128855"]

trvae_epochs = 500
surgery_epochs = 500

early_stopping_kwargs = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

trvae = sca.models.TRVAE(
    adata=source_adata,
    condition_key=condition_key,
    conditions=source_conditions,
    hidden_layer_sizes=[128, 128],
)

trvae.train(
    n_epochs=trvae_epochs,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs
)

adata_latent = sc.AnnData(trvae.get_latent())
adata_latent.obs['cell_type'] = source_adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = source_adata.obs[condition_key].tolist()
sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           )

ref_path = 'reference_model2/'
trvae.save(ref_path, overwrite=True)

new_trvae = sca.models.TRVAE.load_query_data(adata=target_adata, reference_model=ref_path)
new_trvae.train(
    n_epochs=surgery_epochs,
    alpha_epoch_anneal=200,
    early_stopping_kwargs=early_stopping_kwargs,
    weight_decay=0
)
adata_latent = sc.AnnData(new_trvae.get_latent())
adata_latent.obs['cell_type'] = target_adata.obs[cell_type_key].tolist()
adata_latent.obs['batch'] = target_adata.obs[condition_key].tolist()

sc.pp.neighbors(adata_latent, n_neighbors=8)
sc.tl.leiden(adata_latent)
sc.tl.umap(adata_latent)
sc.pl.umap(adata_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           )
full_latent = sc.AnnData(new_trvae.get_latent(adata.X, adata.obs[condition_key]))
full_latent.obs['cell_type'] = adata.obs[cell_type_key].tolist()
full_latent.obs['batch'] = adata.obs[condition_key].tolist()
sc.pp.neighbors(full_latent, n_neighbors=8)
sc.tl.leiden(full_latent)
sc.tl.umap(full_latent)
sc.pl.umap(full_latent,
           color=['batch', 'cell_type'],
           frameon=False,
           wspace=0.6,
           )



