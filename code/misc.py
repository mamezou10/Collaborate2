import numpy as np
import pandas as pd

def transform_exp(adata):
    sc.pp.filter_cells(adata_ref, min_genes=100)
    sc.pp.filter_genes(adata_ref, min_cells=100)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata_ref, n_top_genes=4000 )
    sc.pp.pca(adata_ref)
    sc.pp.neighbors(adata_ref)
    sc.tl.umap(adata_ref)
    sc.tl.tsne(adata_ref)
    sc.pl.tsne(adata_ref, color=["cell_type"], save="kari")
    sc.pl.umap(adata_ref, color=["cell_type"], save="kari")
    return adata



import subprocess
def annotate_geneset(marker_genes, pre="test", geneset_db="/home/hirose/Documents/main/gmts/PanglaoDB_Augmented_2021.txt"):
    n_cluster = len(marker_genes.columns)//2
    karis = list()
    for i in range(n_cluster):
        #import pdb; pdb.set_trace()
        marker_genes_ = marker_genes.iloc[:, (2*i):(2*i+2)]
        marker_genes_.columns =["gene", "pvals"]
        try:
            marker_genes_ = marker_genes_[marker_genes_.pvals < 10**-100].sort_values("pvals", ascending=True).iloc[:100,:]
        except (ValueError, KeyError):
            continue
        num = marker_genes_.shape[0]
        gene = '\t'.join(marker_genes_["gene"].tolist())
        kari = "\t".join(["leiden"+str(i), str(num), gene]) 
        karis.append(kari)
    s = "\n".join(karis)
    with open(pre + "_marker.gmt", "w", encoding="utf-8") as f:
        f.write(s)
    return s
    # command = [
    #     "Rscript", 
    #     "/home/hirose/Documents/main/gmts/for_fisher.r", 
    #     "/home/hirose/Documents/main/gmts/my_fisher.R", 
    #     pre + "_marker.gmt", geneset_db, pre, 
    #     "5"
    # ]
    # subprocess.run(command)
    # enrichment = pd.read_table(pre + "_enrichment.tsv")
    # anno_df = pd.DataFrame()
    # for i in list(np.unique(marker_genes.index)):
    #     data = enrichment[enrichment.geneset=="leiden"+str(i)]['gene_ontology'].str.cat(enrichment[enrichment.geneset=="leiden"+str(i)]['minus_log10_pvalue'].round(2).astype(str), sep='; -log10(p)=')
    #     anno_df = pd.concat([anno_df, pd.DataFrame({"leiden"+str(i): data}).reset_index(drop=True)], axis=1)
    # anno_df.to_csv(pre + "_anno.tsv", sep="\t")
    # return anno_df


from statsmodels.stats.multitest import multipletests
from scipy import stats
def generate_cont_table_batch(top_genes, all_genes, gene_sets):
    S = np.array([
        all_genes.isin(gene_set).astype(int)
        for gene_set in gene_sets.values()
    ])
    t = all_genes.isin(top_genes).astype(int)
    tps = S @ t
    fns = t.sum() - tps
    fps = S.sum(axis=1) - tps
    tns = (all_genes.shape[0] - t.sum()) - fps
    cont_tables = np.array([[tps, fns], [fps, tns]]).transpose((2, 0, 1))
    return cont_tables

def gene_set_enrichment(top_genes, all_genes, gene_sets):
    cont_tables = generate_cont_table_batch(top_genes, all_genes, gene_sets)
    pvals = pd.Series(multipletests([stats.fisher_exact(tb, alternative='two-sided')[1] for tb in cont_tables], method='fdr_bh')[1], index=gene_sets.keys()).sort_values()
    return pvals

def parse_gmt(gmt_fname):
    gene_sets = {}
    with open(gmt_fname) as f:
        for l in f.read().splitlines():
            vals = l.split('\t')
            set_name = vals[0]
            genes = vals[2:]
            gene_sets[set_name] = genes
    return gene_sets