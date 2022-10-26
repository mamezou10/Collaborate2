
import os
import sys
wd = '/mnt/Donald/hoshino/liver/analysis/221016_2'
os.makedirs(wd, exist_ok=True)
os.chdir(wd)

from importlib import reload
import scipy as sp
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import anndata
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import deepcolor
np.random.seed(1)
torch.manual_seed(1)

sys.path.append("/home/hirose/Documents/Collaborate/hoshino/liver/ln_deepcolor_mod")
import workflow as wf

sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/liver/analysis/221016/mapped_sc_adata.h5ad')
sc_adata_raw = sc_adata.copy()
sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/liver/analysis/221016/mapped_sp_adata.h5ad')

sc.pp.neighbors(sc_adata)
sc.tl.umap(sc_adata)
sc.tl.leiden(sc_adata)


sp_adata = wf.calculate_imputed_spatial_expression(sc_adata, sp_adata)
library_names = np.unique(sp_adata.obs["batch"])

#HE保存
fig, axs = plt.subplots(1, 4, figsize=(15,5))
for i in range(len(library_names)):
    sc.pl.spatial(sp_adata[sp_adata.obs.batch == library_names[i],:], library_id=library_names[i],color = None, show=False, ax=axs[i])

fig.tight_layout()
fig.show()
fig.savefig("he.png")

#cluster保存
fig, axs = plt.subplots(1, 4, figsize=(15,5))
for i in range(len(library_names)):
    sc.pl.spatial(sp_adata[sp_adata.obs.batch == library_names[i],:], library_id=library_names[i],color = "leiden", show=False, ax=axs[i])

fig.tight_layout()
fig.show()
fig.savefig("leiden.png")

#sc.pl.umap(sc_adata, color="leiden")

# # for ingest REF
# meta = pd.read_csv("/mnt/Daisy/sc_LIVER_GSE166504/GSE166504_cell_metadata.20220204.tsv.gz", sep="\t")
# count = pd.read_csv("/mnt/Daisy/sc_LIVER_GSE166504/GSE166504_cell_raw_counts.20220204.txt", sep="\t")
# adata_ref = anndata.AnnData(count.T)
# adata_ref.var_names = count.index
# adata_ref.obs_names = count.columns
# samples_df = pd.DataFrame(count.columns).set_index(count.columns)
# meta.index = meta.FileName.str.cat(meta.CellID, sep="_")
# obs = pd.merge(samples_df, meta, how="left", left_index=True, right_index=True)
# adata_ref.obs = obs
# sc.pp.normalize_total(adata_ref)
# sc.pp.log1p(adata_ref)
# sc.pp.highly_variable_genes(adata_ref, n_top_genes=4000,  )
# sc.pp.scale(adata_ref)
# sc.pp.pca(adata_ref)
# sc.pp.neighbors(adata_ref)
# sc.tl.umap(adata_ref)
# sc.tl.tsne(adata_ref)
# sc.pl.tsne(adata_ref, color=["CellType"], save="ref_celltype.png")

# for ingest REF lung
adata_ref = sc.read_h5ad('/mnt/196public/Hoshino/mapped_anndata/ref_adata/sfaira/0b9d8a04-bb9d-44da-aa27-705bb65b54eb/0fb7916e-7a68-4a4c-a441-3ab3989f29a7.h5ad')
adata_ref.obs
sc.pp.filter_cells(adata_ref, min_genes=100)
sc.pp.filter_genes(adata_ref, min_cells=100)
sc.pp.normalize_total(adata_ref)
sc.pp.log1p(adata_ref)
sc.pp.highly_variable_genes(adata_ref, n_top_genes=4000, subset=True )
sc.pp.scale(adata_ref)
sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.umap(adata_ref)
sc.tl.tsne(adata_ref)
sc.pl.tsne(adata_ref, color=["cell_type"], save="ref_lung_celltype.png")
sc.pl.umap(adata_ref, color=["cell_type"], save="ref_lung_celltype.png")

# id convert 
gtf = pd.read_csv("/mnt/Daisy/mouse/gencode.vM25.annotation.gtf", skiprows=5, header=None, sep="\t")
kari=gtf.loc[:,8].str.split("; ", expand=True)
kari=kari.fillna("kari")
for i in range(22):
    kari[i] = kari[i].mask(~kari[i].str.contains("gene_id|gene_name", regex=True))

kari = kari.fillna("")
# df = kari[0]
# for i in range(1,6):
#     df.str.cat(kari[i], sep=", ")
df = pd.DataFrame(kari[0].str.cat([kari[1], kari[2], kari[3], kari[4], kari[5], kari[6], kari[7], kari[8], kari[9], kari[10], 
                    kari[11], kari[12], kari[13], kari[14], kari[15], kari[16], kari[17], kari[18], kari[19], kari[20], kari[21]], sep=","))
df = df[0].str.replace(',| |"|gene_id', "").str.split("gene_name", expand=True)
df.columns = ["gene_id", "gene_name"]
df["gene_id"] = df["gene_id"].str.split(".", expand=True)[0]
df = df.drop_duplicates().reset_index()[["gene_id", "gene_name"]]
df.to_csv("/mnt/Daisy/mouse/id_name.tsv", sep="\t", header=True)

adata_ref.var_names = pd.merge(pd.DataFrame(adata_ref.var_names), df, how="left", left_on="feature_id", right_on="gene_id")["gene_name"]

# for ingest SC
sc_adata_ing = sc_adata_raw.copy()
sc.pp.filter_cells(sc_adata_ing, min_genes=100)
sc.pp.filter_genes(sc_adata_ing, min_cells=100)
sc.pp.normalize_total(sc_adata_ing)
sc.pp.log1p(sc_adata_ing)
sc.pp.highly_variable_genes(sc_adata_ing, n_top_genes=4000, subset=True  )
sc.pp.scale(sc_adata_ing)
sc.pp.pca(sc_adata_ing)
sc.pp.neighbors(sc_adata_ing)
sc.tl.umap(sc_adata_ing)
sc.tl.tsne(sc_adata_ing)
sc.tl.leiden(sc_adata_ing)
sc.pl.tsne(sc_adata_ing, color=["leiden"], save="sc_celltype.png")
sc.pl.tsne(sc_adata_ing, color=None, save="sc_celltype_nocolor.png")

# ingest
var_names = adata_ref.var_names.intersection(sc_adata_ing.var_names)
adata_ref = adata_ref[:, var_names]
sc_adata_ing = sc_adata_ing[:, var_names]
kari_adata = sc.tl.ingest(sc_adata_ing, adata_ref, obs='cell_type', inplace=False)
sc_adata_ing.obs["cell_type"] = kari_adata.obs["cell_type"].copy()
sc.pl.tsne(sc_adata_ing, color=["leiden","cell_type"], 
            legend_loc="on data", legend_fontsize="xx-small", save="ingest_tsne.png")
sc.pl.tsne(sc_adata_ing, color=["cell_type"], save="ingest_tsne_cell_type.png")
sc.pl.umap(sc_adata_ing, color=["leiden","cell_type"], save="ingest_umap.png")

sc_adata_ing.obs[["leiden", "cell_type"]][(sc_adata_ing.obs.leiden=="1")]
sc_adata_ing.obs[["leiden", "cell_type"]][(sc_adata_ing.obs.leiden=="3")]
sc_adata_ing.obs[["leiden", "cell_type"]][(sc_adata_ing.obs.leiden=="27")]
sc_adata_ing.obs[["leiden", "cell_type"]][(sc_adata_ing.obs.leiden=="24")]
sc_adata_ing.obs[["leiden", "cell_type"]][(sc_adata_ing.obs.leiden=="26")]
sc_adata_ing.obs[["leiden", "cell_type"]][(sc_adata_ing.obs.leiden=="18")]



# celltype = pd.DataFrame(sc_adata_ing.obs.leiden)
# celltype["name"]=nan
# celltype.loc[celltype.leiden=="1","name"]="Kupffer cell"
# celltype.loc[celltype.leiden=="4","name"]="Endotherial cell"
# celltype.loc[celltype.leiden=="2","name"]="Hepatocyte"
# celltype.loc[celltype.leiden=="8","name"]="Hepatocyte"
# celltype.loc[celltype.leiden=="5","name"]="Hepatocyte"
# celltype.loc[celltype.leiden=="14","name"]="Hepatocyte"
# celltype.loc[celltype.leiden=="10","name"]="T cell"
# celltype.loc[celltype.leiden=="6","name"]="NK cell"
# celltype.loc[celltype.leiden=="0","name"]="Monocyte/Monocyte derived macrophage"
# celltype.loc[celltype.leiden=="3","name"]="Monocyte/Monocyte derived macrophage"
# celltype.loc[celltype.leiden=="7","name"]="Monocyte/Monocyte derived macrophage"

# celltype_obs = pd.DataFrame(sc_adata_ing.obs)
# celltype_obs["major_celltype"] = celltype.name
# sc_adata_ing.obs = celltype_obs
# sc.pl.tsne(sc_adata_ing, color=["major_celltype"], save="ingest_celltype.png")
# sc.pl.umap(sc_adata_ing, color=["major_celltype"], save="ingest_celltype.png")

# >>> del count
exo_sc_adata = sc.read_h5ad('/mnt/196public/Hoshino/mapped_anndata/exome_sc_adata.h5ad')
exo_sp_adata = sc.read_h5ad('/mnt/196public/Hoshino/mapped_anndata/exome_sp_adata.h5ad')

sc_adata_ing.obs = pd.merge(pd.DataFrame(sc_adata_ing.obs), 
                        pd.DataFrame(exo_sc_adata.obs["exome_coloc"]), 
                        how="left", left_index=True, right_index=True)
sc.pl.tsne(sc_adata_ing, color=["exome_coloc"], save="ingest_exome.png")


lt_df = pd.read_csv('/home/hirose/Documents/Collaborate/gmts/ligand_target_df.csv', index_col=0)
mouse2human = pd.read_csv("/home/hirose/Documents/Collaborate/gmts/mouse2human.txt", sep="\t")
human_genes = pd.merge(pd.DataFrame(sc_adata.var_names), mouse2human, how="left", left_on=0, right_on="mouse")
sc_adata_human = sc_adata.copy()
sc_adata_human.var_names = human_genes.human
sc_adata_human = sc_adata_human[:,[sc_adata_human.var_names[i] is not np.nan for i in list(range(len(sc_adata_human.var_names)))]]
sc_adata_human.var_names_make_unique()
sc_adata_human.var_names_make_unique()
# sc_adata_human.obs_names_make_unique()
np.random.seed(1)
fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human, 'leiden', 
                                                            lt_df, ["1"], 
                                                            celltype_sample_num=1000, ntop_genes=4000, each_display_num=3, role="sender", edge_thresh=0.3)
fig.show()
# fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human[sc_adata_human.obs["exome_coloc"] > np.mean(sc_adata_human.obs["exome_coloc"]),:],         
#                                                             'leiden', lt_df, ["1"], 
#                                                             celltype_sample_num=1000, ntop_genes=4000, each_display_num=3, role="sender", edge_thresh=0.3)
# fig.show()



#sc.pl.spatial(sp_adata[sp_adata.obs.batch == library_names[0],:], library_id=library_names[0],color = "Muc1", show=True)

sc_adata.obsm["X_zl"] = sc_adata.obsm["X_z"].copy()
sc_adata.obsm['p_mat'] = sc_adata.obsm['map2sp'] / np.sum(sc_adata.obsm['map2sp'], axis=1).reshape((-1, 1))

dual_adata = wf.analyze_pair_cluster(sc_adata, sp_adata, ["1"], ["3"], "leiden", max_pair_num=30000)
sc_adata_ing.obs['coloc_cluster'] = 'None'
sc_adata_ing.obs.loc[dual_adata.obs.cell1_obsname, 'coloc_cluster'] = dual_adata.obs.leiden.values
sc_adata_ing.obs.loc[dual_adata.obs.cell2_obsname, 'coloc_cluster'] = dual_adata.obs.leiden.values
# pair_adata = sc_adata_ing[sc_adata_ing.obs['coloc_cluster'] != 'None']
sc.pl.tsne(sc_adata_ing, color=['leiden', 'coloc_cluster'])

# ligand activity
sp_adata_human = sp_adata.copy()
human_genes = pd.merge(pd.DataFrame(sp_adata.var_names), mouse2human, how="left", left_on=0, right_on="mouse")
sp_adata_human.var_names = human_genes.human
sp_adata_human = sp_adata_human[:,[sp_adata_human.var_names[i] is not np.nan for i in list(range(len(sp_adata_human.var_names)))]]
sp_adata_human.var_names_make_unique()
sp_adata_human.var_names_make_unique()

sc.set_figure_params(scanpy=True, fontsize=20)
ligact_adata = wf.add_sp_ligand_activity(sp_adata_human, lt_df, top_fraction = 0.01)

for i in range(len(library_names)):
    wf.plot_sp_ligand_activity(ligact_adata[ligact_adata.obs["batch"]==library_names[i]], gene = 'BMP4', library_id=library_names[i])

for i in range(len(library_names)):
    wf.plot_sp_ligand_activity(ligact_adata[ligact_adata.obs["batch"]==library_names[i]], gene = 'F7', library_id=library_names[i])


# plot corr
plt.figure(figsize=(40,100))
for i in range(len(ligact_adata.var_names)):
    fig_n = len(ligact_adata.var_names)
    gene= ligact_adata.var_names[i]
    print(gene)
    plt.subplot(fig_n//5 + 1, 5, i+1)
    try:
        g, r, p = wf.plot_lig_correlation(ligact_adata, gene = gene, plot=False)
        g
    except KeyError:
        pass

plt.tight_layout()
plt.savefig("corr.png")

df_corr = wf.get_lig_correlation(ligact_adata, ligact_adata.var_names)
pd.DataFrame(df_corr).to_csv("corr.tsv", sep="\t")

# sig_res = wf.get_gaussian_aic(ligact_adata, genes=df_corr[df_corr.r > 0.2].gene.tolist(), thres=[200,500,1000])
sig_res = wf.get_gaussian_aic(ligact_adata, genes=df_corr.gene.tolist(), thres=[200,500,1000])
pd.DataFrame(sig_res).to_csv("sig_res.tsv", sep="\t")

# plot AIC
plt.figure(figsize=(40,100))
for i in range(len(ligact_adata.var_names)):
    fig_n = len(ligact_adata.var_names)
    gene= ligact_adata.var_names[i]
    plt.subplot(fig_n//5 + 1, 5, i+1)
    df = sig_res[sig_res.gene==gene]
    sns.scatterplot(data=df[(df.sig!="no gaussian") & (df.sig!="no X")], x="sig", y="aic", hue="thre")
    plt.scatter(data=df[(df.sig=="no gaussian")], x=[0 for _ in range(len(df[(df.sig=="no gaussian")]))], y="aic", c="red")
    plt.scatter(data=df[(df.sig=="no X")], x=[0 for _ in range(len(df[(df.sig=="no X")]))], y="aic", c="green")
    plt.title(gene)

plt.tight_layout()
plt.savefig("AIC.png")

# plot BIC
plt.figure(figsize=(40,100))
for i in range(len(ligact_adata.var_names)):
    fig_n = len(ligact_adata.var_names)
    gene= ligact_adata.var_names[i]
    plt.subplot(fig_n//5 + 1, 5, i+1)
    df = sig_res[sig_res.gene==gene]
    sns.scatterplot(data=df[(df.sig!="no gaussian") & (df.sig!="no X")], x="sig", y="bic", hue="thre")
    plt.scatter(data=df[(df.sig=="no gaussian")], x=[0 for _ in range(len(df[(df.sig=="no gaussian")]))], y="bic", c="red")
    plt.scatter(data=df[(df.sig=="no X")], x=[0 for _ in range(len(df[(df.sig=="no X")]))], y="bic", c="green")
    plt.title(gene)

plt.tight_layout()
plt.savefig("BIC.png")

## diffusionの確認
## AICが一番小さいところを拾う
res = pd.read_csv("sig_res.tsv", sep="\t")
genes = np.unique(res.gene).tolist()
gene_l = []; sig_l = []; thre_l = []
for i in genes:
    gene_l.append(i)
    sig_l.append(res[res.gene==i].iloc[np.argmin(res[res.gene==i].bic),].sig )
    thre_l.append(res[res.gene==i].iloc[np.argmin(res[res.gene==i].bic),].thre )

sigs=pd.DataFrame({"gene":gene_l, "sig":sig_l, "thre":thre_l})   

for gene in ["BMP4", "F7", "SEMA3B", "SLIT2", "MAPT"]:
    fig = plt.figure(figsize=(20,10))
    fig.add_subplot(1, 2, 1)
    wf.plot_lig_correlation(ligact_adata, gene = gene, plot=False)
    fig.add_subplot(1, 2, 2) 
    edges_=wf.make_edges_dist(ligact_adata, thres=int(sigs[sigs.gene==gene].thre), gene=gene)
    ga_coor = wf.add_gaussian(edges_, sig=int(sigs[sigs.gene==gene].sig), adata=ligact_adata)
    plt.ylabel("ligand activity")
    plt.xlabel("sigma:"+str(int(sigs[sigs.gene==gene].sig)))
    plt.xlabel("sigma:"+str(int(sigs[sigs.gene==gene].sig)))
    sig = str(int(sigs[sigs.gene==gene].sig))
    wf.plot_lig_correlation2(ligact_adata, ga_coor, gene = gene, sigma=sig, plot=False)
    plt.savefig("dissusion_" + gene + ".png")




## 各サンプルごと
library_names = np.unique(sp_adata.obs["batch"])
for library in library_names:
    sc.set_figure_params(scanpy=True, fontsize=20)
    ligact_adata = wf.add_sp_ligand_activity(sp_adata_human[sp_adata_human.obs["batch"]==library], lt_df, top_fraction = 0.01)
    # plot corr
    plt.figure(figsize=(40,100))
    for i in range(len(ligact_adata.var_names)):
        fig_n = len(ligact_adata.var_names)
        gene= ligact_adata.var_names[i]
        print(gene)
        plt.subplot(fig_n//5 + 1, 5, i+1)
        try:
            g, r, p = wf.plot_lig_correlation(ligact_adata, gene = gene, plot=False)
            g
        except KeyError:
            pass
    plt.tight_layout()
    plt.savefig("corr_"+library+".png")
    df_corr = wf.get_lig_correlation(ligact_adata, ligact_adata.var_names)
    pd.DataFrame(df_corr).to_csv("corr_"+library+".tsv", sep="\t")
    # sig_res = wf.get_gaussian_aic(ligact_adata, genes=df_corr[df_corr.r > 0.2].gene.tolist(), thres=[200,500,1000])
    sig_res = wf.get_gaussian_aic(ligact_adata, genes=df_corr.gene.tolist(), thres=[200,500,1000])
    pd.DataFrame(sig_res).to_csv("sig_res_"+library+".tsv", sep="\t")
    # plot AIC
    plt.figure(figsize=(40,100))
    for i in range(len(ligact_adata.var_names)):
        fig_n = len(ligact_adata.var_names)
        gene= ligact_adata.var_names[i]
        plt.subplot(fig_n//5 + 1, 5, i+1)
        df = sig_res[sig_res.gene==gene]
        sns.scatterplot(data=df[(df.sig!="no gaussian") & (df.sig!="no X")], x="sig", y="aic", hue="thre")
        plt.scatter(data=df[(df.sig=="no gaussian")], x=[0 for _ in range(len(df[(df.sig=="no gaussian")]))], y="aic", c="red")
        plt.scatter(data=df[(df.sig=="no X")], x=[0 for _ in range(len(df[(df.sig=="no X")]))], y="aic", c="green")
        plt.title(gene)
    plt.tight_layout()
    plt.savefig("AIC_"+library+".png")
    # plot BIC
    plt.figure(figsize=(40,100))
    for i in range(len(ligact_adata.var_names)):
        fig_n = len(ligact_adata.var_names)
        gene= ligact_adata.var_names[i]
        plt.subplot(fig_n//5 + 1, 5, i+1)
        df = sig_res[sig_res.gene==gene]
        sns.scatterplot(data=df[(df.sig!="no gaussian") & (df.sig!="no X")], x="sig", y="bic", hue="thre")
        plt.scatter(data=df[(df.sig=="no gaussian")], x=[0 for _ in range(len(df[(df.sig=="no gaussian")]))], y="bic", c="red")
        plt.scatter(data=df[(df.sig=="no X")], x=[0 for _ in range(len(df[(df.sig=="no X")]))], y="bic", c="green")
        plt.title(gene)
    plt.tight_layout()
    plt.savefig("BIC_"+library+".png")

plt.close("all")
for library in library_names:
    print(library)
    #os.mkdir("dissusion_" + library)
    ligact_adata = wf.add_sp_ligand_activity(sp_adata_human[sp_adata_human.obs["batch"]==library], lt_df, top_fraction = 0.01)
    res = pd.read_csv("sig_res_"+library+".tsv", sep="\t")
    genes = np.unique(res.gene).tolist()
    gene_l = []; sig_l = []; thre_l = []
    for i in genes:
        gene_l.append(i)
        sig_l.append(res[res.gene==i].iloc[np.argmin(res[res.gene==i].bic),].sig )
        thre_l.append(res[res.gene==i].iloc[np.argmin(res[res.gene==i].bic),].thre )
    sigs=pd.DataFrame({"gene":gene_l, "sig":sig_l, "thre":thre_l})  
    print(sigs)
    for gene in sigs.gene:
        sig = sigs[sigs.gene==gene].iloc[0,1]
        if sig=="no gaussian":
            print("no")
        elif sig=="no X":
            print("no")
        else:
            print(gene)
            fig = plt.figure(figsize=(20,10))
            fig.add_subplot(1, 2, 1)
            wf.plot_lig_correlation(ligact_adata, gene = gene, plot=False)
            fig.add_subplot(1, 2, 2) 
            edges_=wf.make_edges_dist(ligact_adata, thres=int(sigs[sigs.gene==gene].thre), gene=gene)
            ga_coor = wf.add_gaussian(edges_, sig=int(sigs[sigs.gene==gene].sig), adata=ligact_adata)
            plt.ylabel("ligand activity")
            plt.xlabel("sigma:"+str(int(sigs[sigs.gene==gene].sig)))
            plt.xlabel("sigma:"+str(int(sigs[sigs.gene==gene].sig)))
            sig = str(int(sigs[sigs.gene==gene].sig))
            wf.plot_lig_correlation2(ligact_adata, ga_coor, gene = gene, sigma=sig, plot=False)
            plt.savefig("dissusion_" + library + "/" + gene + ".png")
            fig = plt.figure(figsize=(20,10))
            wf.plot_sp_ligand_activity(ligact_adata, gene = gene, library_id=library)
            plt.savefig("dissusion_" + library + "/" + gene + "_spatial.png")






exo_sc_adata = sc.read_h5ad('/mnt/196public/Hoshino/mapped_anndata/exome_sc_adata.h5ad')
exo_sp_adata = sc.read_h5ad('/mnt/196public/Hoshino/mapped_anndata/exome_sp_adata.h5ad')


gene="Bmp4"

kari = pd.merge(pd.DataFrame(sp_adata.obs), pd.DataFrame(exo_sp_adata.obs["exome"]), how="left", left_index=True, right_index=True)
x = pd.DataFrame(kari["exome"]).reset_index()["exome"]
y = pd.DataFrame(sp_adata[:,gene].layers["imputed_exp"]); y.columns = ["expression"]
# x = pd.DataFrame(exo_sp_adata[:,gene].X)
df = pd.concat([x,y], axis=1)
df.columns=["x","y"]
df = df.dropna()
r, p = sp.stats.pearsonr(x=df.x, y=df.y)
fig = plt.figure(figsize=(10,10))
g = sns.regplot(x=x, y=y, data=df)
anc = AnchoredText("r: "+str(round(r,3))+", p: "+str(p), loc="upper left", frameon=False, prop=dict(size=15))
g.axes.add_artist(anc)
g.set_title(gene)
plt.savefig("kari.png")


sp_adata.obs = pd.merge(pd.DataFrame(sp_adata.obs), 
                        pd.DataFrame(exo_sp_adata.obs["exome"]), 
                        how="left", left_index=True, right_index=True)

sc_adata.obs = pd.merge(pd.DataFrame(sc_adata.obs), 
                        pd.DataFrame(exo_sc_adata.obs["exome_coloc"]), 
                        how="left", left_index=True, right_index=True)

sc.pl.tsne(sc_adata, color="exome", save="exome.png")




