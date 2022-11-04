
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221031/'
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

sys.path.append("/home/hirose/Documents/main/Collaborate2/hoshino/kidney")
import deepcolor_mod.workflow as wf


sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221024/adata_total_annotated.h5')
sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/sp_adata.h5ad')

sc_adata = sc_adata[:, sc_adata.layers['count'].toarray().sum(axis=0) > 10]
sp_adata = sp_adata[:, sp_adata.layers['count'].toarray().sum(axis=0) > 10]
common_genes = np.intersect1d(sc_adata.var_names, sp_adata.var_names)
sc_adata = sc_adata[:, common_genes]
sp_adata = sp_adata[:, common_genes]

reload(wf)
sc_adata, sp_adata = wf.estimate_spatial_distribution(sc_adata, sp_adata, 
                            x_batch_size=300, s_batch_size=300, 
                            param_save_path='/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/opt_params.pt', 
                            first_epoch=1000, second_epoch=1000)

# zでumap与え直し
sc.pp.neighbors(sc_adata, use_rep="X_zl")
sc.tl.umap(sc_adata)
sc.pl.umap(sc_adata, color="Annotation")

sp_adata = wf.calculate_imputed_spatial_expression(sc_adata, sp_adata)

sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color="Aadat", use_raw=False)
sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color="Aadat", layer="imputed_exp")
sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color="Aadat", layer="count")

"Dock5" in sp_adata.var_names



sp_adata = wf.calculate_clusterwise_distribution(sc_adata, sp_adata, 'Annotation')

celltypes = np.unique(sc_adata.obs["Annotation"])
library_names = ["PBS","CTL","PE2","PE1"]
for celltype in celltypes:
    fig = plt.figure(figsize=(20,5))
    for i, library in enumerate(library_names):
        axes = fig.add_subplot(1,4,i+1)
        sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]==library, :], library_id=library, 
                    color=celltype, ax=axes, show=False, title="")
    plt.suptitle(celltype)
    plt.savefig(f"figures/spatial_{celltype}.png")




library_names = np.unique(sp_adata[sp_adata.obs["library_id"]=="PBS", :], library_id="PBS")

#HE保存
fig, axs = plt.subplots(1, 4, figsize=(15,5))
for i in range(len(library_names)):
    sc.pl.spatial(sp_adata[sp_adata.obs.batch == library_names[i],:], library_id=library_names[i],color = None, show=False, ax=axs[i])

fig.tight_layout()
fig.show()
fig.savefig("he.png")





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




