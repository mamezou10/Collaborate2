
import scanpy as sc
import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy.sparse import coo_matrix
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from numpy import mean
import statannot
import matplotlib.pyplot as plt
wd = "/mnt/Bambi/Projects/morim_resist/analysis/220811_pattern2/"
os.makedirs(wd, exist_ok=True)
os.chdir(wd)

## MUC1>0を元データから取得
raw_adata = sc.read_h5ad("/mnt/Bambi/Projects/yao/Lee_2021/sc_crc.h5ad")
bool = pd.DataFrame(raw_adata[:,raw_adata.var_names=="MUC1"].X)>0
MUC1pos_cells = pd.DataFrame(raw_adata.obs_names)[bool.values.reshape(-1)].values.reshape(-1).tolist()

bool = pd.DataFrame(raw_adata[:,raw_adata.var_names=="MICA"].X)>0
MICApos_cells = pd.DataFrame(raw_adata.obs_names)[bool.values.reshape(-1)].values.reshape(-1).tolist()


# pattern2のpreprocessed data
adata = sc.read_h5ad("/mnt/Bambi/scCRC_Lee/scCRC.h5ad")
sc.pl.tsne(adata, color="Cell_type", legend_loc="on data", save="tsne.pdf")
sc.pl.umap(adata, color="Cell_type", legend_loc="on data", save="umap.pdf")
adata = adata[adata.obs.Class=="Tumor"]
adata = adata[adata.obs.Cell_type=="Epithelial cells"]

means = pd.concat([pd.Series(adata.obs.Patient), pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)], axis=1)

means_df = means.iloc[:, :1]
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


sns.clustermap(means_df[["MUC1","MICA"]], z_score=1, vmax=3, yticklabels=False, col_cluster=False, linewidths=0, rasterized=True, cmap="coolwarm")# ; plt.show() #‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
plt.savefig("heatmap_MICA.pdf")
sns.clustermap(means_df[["MUC1","MICB"]], z_score=1, vmax=3, yticklabels=False, col_cluster=False, linewidths=0, rasterized=True, cmap="coolwarm")
plt.savefig("heatmap_MICB.pdf")
sns.clustermap(means_df[["MUC1","MICA", "MICB"]], z_score=1, vmax=3, yticklabels=False, col_cluster=False, linewidths=0, rasterized=True, cmap="coolwarm")
plt.savefig("heatmap_MICAB.pdf")


## add heatmap modifyed
df_raw = means_df[["MUC1","MICA"]]
df_raw[df_raw <0] = 0
df_raw = df_raw.sort_values(["MICA", "MUC1"], ascending=False)
sns.clustermap(df_raw, z_score=1, vmin=0, vmax=3, yticklabels=False, col_cluster=False, row_cluster=True, linewidths=0.0, rasterized=True, cmap="coolwarm")# ; plt.show() 
plt.savefig("heatmap_MICA_0adj.pdf")
sns.clustermap(df_raw, z_score=1, vmin=0, vmax=3, yticklabels=False, col_cluster=False, row_cluster=False, linewidths=0.0, rasterized=True, cmap="coolwarm")# ; plt.show() 
plt.savefig("heatmap_MICA_0adj_noclust.pdf")

df_raw = means_df[["MUC1","MICB"]]
df_raw[df_raw <0] = 0
df_raw = df_raw.sort_values(["MICB", "MUC1"], ascending=False)
sns.clustermap(df_raw, z_score=1, vmin=0, vmax=3, yticklabels=False, col_cluster=False, row_cluster=True, linewidths=0.0, rasterized=True, cmap="coolwarm")# ; plt.show() 
plt.savefig("heatmap_MICB_0adj.pdf")
sns.clustermap(df_raw, z_score=1, vmin=0, vmax=3, yticklabels=False, col_cluster=False, row_cluster=False, linewidths=0.0, rasterized=True, cmap="coolwarm")# ; plt.show() 
plt.savefig("heatmap_MICB_0adj_noclust.pdf")



# numeric_columns = means_df.select_dtypes(include=['number']).columns
# means_df[numeric_columns] = means_df[numeric_columns].fillna(0)
means_grouped = means_df.groupby(["Patient"]).mean()
means_grouped = means_grouped.reset_index()

fig_df_total = means_grouped
n=1
fig = plt.figure(figsize=(15,13))
for gene in ["MICA","MICB"]: # ,"GZMB", "PRF1", "TNF", "TNFSF10", "FASLG", "IFNG"
    #gene="MICB"
    fig_df = fig_df_total[["Patient", "MUC1", gene]].dropna()
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
MUC1low_patients = means_grouped.loc[means_grouped.MUC1 <= means_grouped.MUC1.median()]["Patient"].values.reshape(-1)


high_group = np.array(means_df.query('Patient not in @MUC1low_patients')["MICA"])
low_group  = np.array(means_df.query('Patient in @MUC1low_patients')["MICA"])
high_group_micb = np.array(means_df.query('Patient not in @MUC1low_patients')["MICB"])
low_group_micb  = np.array(means_df.query('Patient in @MUC1low_patients')["MICB"])

df = pd.DataFrame({
    "MUC1_status": ["high_patients" for _ in range(len(high_group))] + ["low_patients" for _ in range(len(low_group))],
    "MICA": np.hstack((high_group, low_group)),
    "MICB": np.hstack((high_group_micb, low_group_micb))
})


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
ax = sns.barplot(data=df, y="MICA", x="MUC1_status", errwidth=1, capsize=0.1)
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
plt.savefig("bar_MICA.pdf")

plt.close('all')
ax = sns.barplot(data=df, y="MICB", x="MUC1_status", errwidth=1, capsize=0.1)
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
plt.savefig("bar_MICB.pdf")

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



plt.close('all')
ax = sns.boxplot(data=df, y="MICA", x="MUC1_status", width=0.3, linewidth=0.2, sym="", # whis=50,
                 showmeans=True, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"})#, order=["low_patients", "high_patients"])
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
# plt.show()
plt.savefig("box_MICA.pdf")


plt.close('all')
ax = sns.violinplot(data=df, y="MICA", x="MUC1_status", width=0.9, scale="count", linewidth=0.2)#, order=["low_patients", "high_patients"])
ax = sns.pointplot(data=df, y="MICA", x="MUC1_status", estimator=mean, color="black", ci=None, linestyles="", markers="+")

# ax = sns.stripplot(data=df, y="MICA", x="MUC1_status",color="black", size=2, alpha=0.7)#, order=["low_patients", "high_patients"])
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
plt.savefig("violin_MICA_mean.pdf")
