
import scanpy as sc
import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy.sparse import coo_matrix
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

import statannot
wd = "/mnt/Bambi/Projects/morim_resist/analysis/220805_2/"
os.makedirs(wd, exist_ok=True)
os.chdir(wd)

# MUC1>0で前処理
sc_adata = sc.read_10x_h5("/mnt/Bambi/scCRC_GSE178341/GSE178341_crc10x_full_c295v4_submit.h5")
sc_adata.var_names_make_unique()
bool = pd.DataFrame.sparse.from_spmatrix(sc_adata[:,sc_adata.var_names=="MUC1"].X) > 0
 
sc_adata = sc_adata[bool.values.reshape(-1)]

sc.pp.filter_cells(sc_adata, min_genes=200)
sc.pp.filter_genes(sc_adata, min_cells=3)
sc_adata.var['mt'] = sc_adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(sc_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc_adata = sc_adata[sc_adata.obs.n_genes_by_counts < 2500, :]
sc_adata = sc_adata[sc_adata.obs.pct_counts_mt < 6, :]
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
sc.pp.highly_variable_genes(sc_adata, n_top_genes=2000)

sc.pp.regress_out(sc_adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(sc_adata, max_value=10)

meta_data = pd.read_csv("/mnt/Bambi/scCRC_GSE178341/GSE178341_crc10x_full_c295v4_submit_metatables.csv")
meta_data_=meta_data.set_index("cellID")
sc_adata.obs = pd.merge(pd.DataFrame(sc_adata.obs), meta_data_, how="left", left_index=True, right_index=True)

cluster_data = pd.read_csv("/mnt/Bambi/scCRC_GSE178341/GSE178341_crc10x_full_c295v4_submit_cluster.csv")
cluster_data_=cluster_data.set_index("sampleID")
sc_adata.obs = pd.merge(pd.DataFrame(sc_adata.obs), cluster_data_, how="left", left_index=True, right_index=True)

sc.tl.pca(sc_adata, svd_solver='arpack')
sc.tl.tsne(sc_adata, n_pcs = 30)
sc.pp.neighbors(sc_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(sc_adata)
sc.tl.leiden(sc_adata)
sc.tl.louvain(sc_adata) 



# pattern3のpreprocessed data②
adata = sc_adata
sc.pl.tsne(adata, color="clTopLevel", legend_loc="on data", save="tsne.pdf")
adata = adata[adata.obs.SPECIMEN_TYPE=="T"]
# adata = adata[adata.obs.cl295v11SubShort.isin(["cE01","cE02","cE03","cE04","cE05"])]
adata = adata[adata.obs.clTopLevel=="Epi"]


means = pd.concat([pd.Series(adata.obs.PID), pd.Series(adata.obs.clTopLevel), pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)], axis=1)

means_df = means.iloc[:, :2]
means_df["MUC1"] = np.array(means.MUC1)
means_df["MICA"] = np.array(means.MICA)
means_df["MICB"] = np.array(means.MICB)
# means_df["GZMB"] = np.array(means.GZMB)
# means_df["PRF1"] = np.array(means.PRF1)
# means_df["TNF"] = np.array(means.TNF)
# means_df["TNFSF10"] = np.array(means.TNFSF10)
# means_df["FASLG"] = np.array(means.FASLG)
# means_df["IFNG"] = np.array(means.IFNG)
means_df = means_df.query('index == @MUC1pos_cells')
# means_df = means_df.query('index == @MICApos_cells')

sns.clustermap(means_df[["MUC1","MICA"]], z_score=1, vmax=10, yticklabels=False, col_cluster=False)
plt.savefig("heatmap_MICA.pdf")
sns.clustermap(means_df[["MUC1","MICB"]], z_score=1, vmax=10, yticklabels=False, col_cluster=False)
plt.savefig("heatmap_MICB.pdf")
sns.clustermap(means_df[["MUC1","MICA", "MICB"]], z_score=1, vmax=10, yticklabels=False, col_cluster=False)
plt.savefig("heatmap_MICAB.pdf")


# numeric_columns = means_df.select_dtypes(include=['number']).columns
# means_df[numeric_columns] = means_df[numeric_columns].fillna(0)
means_grouped = means_df.groupby(["PID"]).mean()
means_grouped = means_grouped.reset_index()

fig_df_total = means_grouped
n=1
fig = plt.figure(figsize=(15,13))
for gene in ["MICA","MICB"]: # ,"GZMB", "PRF1", "TNF", "TNFSF10", "FASLG", "IFNG"
    #gene="MICB"
    fig_df = fig_df_total[["PID", "MUC1", gene]].dropna()
    axes = fig.add_subplot(3,3, n)
    r, p = sp.stats.spearmanr(fig_df.MUC1, fig_df[gene])
    # r, p = sp.stats.pearsonr(x=fig_df.MUC1, y=fig_df[gene])
    g = sns.regplot(data=fig_df, x="MUC1", y=gene, scatter_kws={'s':0}, ci=None)# , robust=True)
    g = sns.scatterplot(data=fig_df, x="MUC1", y=gene)
    anc = AnchoredText("r: "+str(round(r,3))+", p: "+str(round(p,3)), loc="upper left", frameon=False, prop=dict(size=15))
    g.axes.add_artist(anc)
    n += 1


fig.tight_layout()
fig.savefig("scatter_MICAB.png")


# means_df["MUC1_status"] = "high"
# means_df.loc[means_df.MUC1 < means_df.MUC1.median(), "MUC1_status"] = "low"
MUC1low_patients = means_grouped.loc[means_grouped.MUC1 < means_grouped.MUC1.median()]["PID"].values.reshape(-1)


high_group = np.array(means_df.query('PID not in @MUC1low_patients')["MICA"])
low_group  = np.array(means_df.query('PID in @MUC1low_patients')["MICA"])
high_group_micb = np.array(means_df.query('PID not in @MUC1low_patients')["MICB"])
low_group_micb  = np.array(means_df.query('PID in @MUC1low_patients')["MICB"])

df = pd.DataFrame({
    "MUC1_status": ["high_patients" for _ in range(len(high_group))] + ["low_patients" for _ in range(len(low_group))],
    "MICA": np.hstack((high_group, low_group)),
    "MICB": np.hstack((high_group_micb, low_group_micb))
})

# low_group  = np.array(means_df.MICA[means_df.MUC1_status=="low"])
# high_group = np.array(means_df.MICA[means_df.MUC1_status=="high"])



# stats.mannwhitneyu(high_group[~np.isnan(high_group)], low_group[~np.isnan(low_group)], alternative="greater")
# stats.ttest_ind(high_group[~np.isnan(high_group)], low_group[~np.isnan(low_group)], equal_var = False)


plt.close('all')
ax = sns.violinplot(data=df, y="MICA", x="MUC1_status", width=0.9, scale="count", linewidth=0.2)#, order=["low_patients", "high_patients"])
ax = sns.stripplot(data=df, y="MICA", x="MUC1_status",color="black", size=2, alpha=0.7)#, order=["low_patients", "high_patients"])
# ax = sns.stripplot(data=df, y="MICA", x="MUC1_status",color="black", size=2, alpha=0.7, jitter=0.01)#, order=["low_patients", "high_patients"])
statannot.add_stat_annotation(
    ax,
    data=df, y="MICA", x="MUC1_status",
    box_pairs=[
        (("low_patients", "high_patients")),
    ],
    test="Mann-Whitney-ls",
    text_format="full",
    loc="outside",
)
plt.savefig("violin_MICA.pdf")


plt.close('all')
ax = sns.violinplot(data=df, y="MICB", x="MUC1_status", width=0.9, scale="count", linewidth=0.2)#, order=["low_patients", "high_patients"])
ax = sns.stripplot(data=df, y="MICB", x="MUC1_status",color="black", size=2, alpha=0.7)#, order=["low_patients", "high_patients"])
# ax = sns.stripplot(data=df, y="MICB", x="MUC1_status",color="black", size=2, alpha=0.7, jitter=0.01)#, order=["low_patients", "high_patients"])
statannot.add_stat_annotation(
    ax,
    data=df, y="MICB", x="MUC1_status",
    box_pairs=[
        (("low_patients", "high_patients")),
    ],
    test="Mann-Whitney-ls",
    text_format="full",
    loc="outside",
)
plt.savefig("violin_MICB.pdf")



