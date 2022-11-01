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