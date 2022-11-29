import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os


out_dir = "/mnt/Donald/tsuji/mouse_sc/221124/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

data_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
adata_mouse=sc.read_h5ad(f'{data_dir}/adata_total_cellClass1.h5')

mcherry_d4 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day4_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day7_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_d10 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day10_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_KO_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_KO_day7_mCherry/outs/filtered_feature_bc_matrix.h5")

library_names = sampleNames = ["Day4", "Day7", "Day10", "KO_Day7"]

adata = mcherry_d4.concatenate(
    [mcherry_d7, mcherry_d10, mcherry_KO_d7],
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=library_names
)


# mCherryを載せる
mcherry = pd.DataFrame.sparse.from_spmatrix(adata.X)
mcherry.columns = ['test1', 'mCherry']
mcherry.index = [str[:16] for str in adata.obs_names.tolist()]

adata_mouse.obs_names = [str[:16] for str in adata_mouse.obs_names.tolist()]


adata_mouse.obs = pd.merge(pd.DataFrame(adata_mouse.obs), mcherry, how="left", left_index=True, right_index=True).fillna({"mCherry": 0})
ids = adata_mouse.obs_names[adata_mouse.obs["mCherry"]>0]
ax = sc.pl.umap(adata_mouse, size=100, show=False)
sc.pl.umap(
    adata_mouse[adata_mouse.obs_names.isin(ids),:],
    size=100,
    color="mCherry",
    ax=ax,
    save="total_mCherry.png"
)


# >>> adata_mouse[adata_mouse.obs_names.isin(ids),:].obs
#                   n_genes  n_genes_by_counts  total_counts  total_counts_mt  pct_counts_mt leiden   sample batch clusters coarse_leiden    cellClass1  test1  mCherry  mCherry_pos
# ACACCAACACGCCAGT   4111.0             4109.0       21804.0            370.0       1.696936      4    Day10     0        5             2  cellClass1-1    NaN    242.0         True
# ACTTTCAAGAGCTATA   1789.0             1789.0        7514.0             75.0       0.998137     22     Day7     0       18             7  cellClass1-1    NaN     60.0         True
# AGCGTCGAGAGTTGGC   4371.0             4370.0       13968.0             97.0       0.694444     22     Day7     0       18             7  cellClass1-1    NaN     90.0         True
# AGCTTGACATCCAACA   4297.0             4297.0       13920.0            314.0       2.255747     10     Day4     0        8             2  cellClass1-1    NaN    231.0         True
# CAGCCGAGTATGGTTC   2158.0             2158.0        4918.0             74.0       1.504677      3     Day4     0        3             3  cellClass1-1    NaN     57.0         True
# CAGGTGCGTCGTGGCT   3792.0             3792.0       14588.0            110.0       0.754044     22     Day7     0       18             7  cellClass1-1    NaN     72.0         True
# CATCCACTCGTGTAGT   4145.0             4145.0       15181.0             59.0       0.388644     22     Day4     0       18             7  cellClass1-1    NaN     67.0         True
# CCAGCGAGTAAACACA   1949.0             1949.0        3982.0             55.0       1.381215      9  KO_Day7     1        7             5  cellClass1-3    NaN    144.0         True
# CCTACCAAGCAACGGT    840.0              840.0       13130.0              0.0       0.000000     14    Day10     0       14             7  cellClass1-1    NaN     62.0         True
# CGATCGGGTGGCGAAT   4216.0             4216.0       14547.0            210.0       1.443597     22    Day10     0       18             7  cellClass1-1    NaN     50.0         True
# CGTAGCGTCGCACTCT   1421.0             1421.0        3045.0             51.0       1.674877      1     Day7     0        2             0  cellClass1-1    NaN    167.0         True
# CGTCAGGAGATCCTGT   1839.0             1839.0       17217.0            599.0       3.479119     22     Day4     0       18             7  cellClass1-1    NaN     59.0         True
# CTCACACCATCGATGT   4391.0             4390.0       22035.0            465.0       2.110279     22     Day7     0       18             7  cellClass1-1    NaN     82.0         True
# CTGCCTAAGACCACGA   3436.0             3436.0       10599.0            168.0       1.585055      6  KO_Day7     1        4             0  cellClass1-1    NaN    306.0         True
# GATCGCGTCAAACCGT   1821.0             1821.0        4642.0             47.0       1.012495      1     Day7     0        1             0  cellClass1-1    NaN    103.0         True
# GATTCAGCACGAAGCA   3793.0             3793.0       10596.0            321.0       3.029445      9  KO_Day7     1        7             5  cellClass1-3    NaN     62.0         True
# GCAATCACACATCCAA   4498.0             4498.0       14158.0            391.0       2.761690     22    Day10     0       18             7  cellClass1-1    NaN     65.0         True
# GCAGCCAAGTCCAGGA   4448.0             4448.0       14227.0            400.0       2.811556     22    Day10     0       18             7  cellClass1-1    NaN     53.0         True
# GCTGCAGTCTTGTACT   4551.0             4549.0       51552.0            817.0       1.584808     22     Day4     0       18             7  cellClass1-1    NaN    130.0         True
# GGTGAAGAGCCACGCT   2073.0             2073.0        5998.0             73.0       1.217072      3     Day4     0        3             3  cellClass1-1    NaN     77.0         True
# GTCCTCATCACGGTTA   4714.0             4714.0       17989.0            171.0       0.950581     22     Day7     0       18             7  cellClass1-1    NaN    173.0         True
# GTGGGTCCACCCATGG   2607.0             2607.0        6473.0            124.0       1.915650      8    Day10     0       11             0  cellClass1-1    NaN    382.0         True
# TGACAACTCAAACAAG   2988.0             2986.0       10574.0              2.0       0.018914     22     Day4     0       18             7  cellClass1-1    NaN     78.0         True
# TTAGTTCCAAACGTGG   3222.0             3220.0       38500.0           1055.0       2.740260     22     Day4     0       18             7  cellClass1-1    NaN    172.0         True
# TTAGTTCCAAACGTGG   2462.0             2462.0        5570.0             52.0       0.933573      0     Day7     0        0             1  cellClass1-1    NaN    172.0         True

mCherryposが586細胞 -> 25/14665 ほとんどfilterで消えてる

