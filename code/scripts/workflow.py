from threading import enumerate
from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import scvelo as scv
import joblib
import os
import anndata
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import sys
sys.path.append('envdyn/')

sys.path.append('scripts/')

from scripts import commons
import umap
from scripts import utils

import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colors
import importlib
import torch
from scripts import basic
import envdyn
import json
import pytorch_lightning as pl
import dataset
import modules
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

class tqdm(_tqdm):
    """
    Custom tqdm progressbar where we append 0 to floating points/strings to prevent the progress bar from flickering
    """

class FixTqdmProgress(ProgressBar):
    def init_sanity_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar


    def init_train_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = tqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=80,
            file=sys.stdout,
            smoothing=0,
        )
        return bar


    def init_predict_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=80,
            file=sys.stdout,
            smoothing=0,
        )
        return bar


    def init_validation_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            ncols=80,
            file=sys.stdout,
        )
        return bar


    def init_test_tqdm(self) -> tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            ncols=80,
            file=sys.stdout,
        )
        return bar


def make_norm_mat(s, ratio=1.0):
    snorm_mat = torch.mean(s, dim=1, keepdim=True) * torch.mean(s, dim=0, keepdim=True)
    # snorm_mat = torch.mean(s, dim=0, keepdim=True) * torch.ones_like(s)
    snorm_mat = ratio * torch.mean(s) * snorm_mat / torch.mean(snorm_mat)
    return snorm_mat

def make_datasets(adata, condition_key='condition', batch_key='sample', cond_in_obsm=False):
    if type(adata.layers['spliced']) == np.ndarray:
        s = torch.tensor(adata.layers['spliced'].astype(int)).float()
        u = torch.tensor(adata.layers['unspliced'].astype(int)).float()
    else:
        s = torch.tensor(adata.layers['spliced'].toarray().astype(int)).float()
        u = torch.tensor(adata.layers['unspliced'].toarray().astype(int)).float()
    u[:, ~adata.var.highly_variable] = 0
    snorm_mat = make_norm_mat(s)
    unorm_mat = make_norm_mat(u)
    adata.obs['batch'] = adata.obs[batch_key]
    b, adata = commons.make_sample_one_hot_mat(adata, sample_key=batch_key)
    if cond_in_obsm:
        t = torch.tensor(adata.obsm[condition_key])
    else:
        t, adata = commons.make_sample_one_hot_mat(adata, sample_key=condition_key, stored_obsm_key=condition_key)
    return s, u, snorm_mat, unorm_mat, b, t, adata


def optimize_vdicdyf(train_ds, val_ds, model_params, checkpoint_dirname, module, epoch=1000, lr=0.003, patience=100, batch_size=128, two_step=False, dyn_mode=True, only_vae=False):
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=6, pin_memory=True)
    lit_envdyn = module(**model_params)
    checkpoint_callback = ModelCheckpoint(checkpoint_dirname)
    bar = FixTqdmProgress()
    trainer = pl.Trainer(max_epochs=epoch, gpus=1, callbacks=[bar, EarlyStopping(monitor="val_loss", patience=patience), checkpoint_callback])
    if two_step:
        lit_envdyn.vae_mode()
        trainer.fit(lit_envdyn, train_loader, val_loader)
        lit_envdyn = module.load_from_checkpoint(**model_params, checkpoint_path=checkpoint_callback.best_model_path)
        lit_envdyn.configure_optimizers()
        checkpoint_callback = ModelCheckpoint(checkpoint_dirname)
        # lit_envdyn.dyn_mode()
        trainer = pl.Trainer(max_epochs=epoch, gpus=1, callbacks=[bar, EarlyStopping(monitor="val_loss", patience=patience), checkpoint_callback])
        if dyn_mode:
            lit_envdyn.dyn_mode()
    elif only_vae:
        lit_envdyn.vae_mode()
    trainer.fit(lit_envdyn, train_loader, val_loader)
    lit_envdyn = module.load_from_checkpoint(**model_params, checkpoint_path=checkpoint_callback.best_model_path)
    lit_envdyn.eval()
    return lit_envdyn


def postprocess_civcdyf(adata, model, use_highly_variable=True):
    if use_highly_variable:
        adata = adata[:, adata.var.highly_variable]
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata)
    ds = dataset.CvicDyfDataSet(s, u, snorm_mat, unorm_mat, b, t)
    adata = commons.update_latent(adata, model, ds)
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata = anndata.concat([
        commons.update_dembed_full(
            adata[idx], model, ds[idx], sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1, fix_embed=True)
        for idx in idx_list], merge='unique')
    # adata = commons.update_dembed_full(adata, model, ds, sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1)
    return adata


def modify_model_params(model_params, adata, condition_key='condition', batch_key='sample', cond_in_obsm=True):
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata, batch_key=batch_key, condition_key=condition_key, cond_in_obsm=cond_in_obsm)
    model_params['x_dim'] = s.size()[1]
    model_params['batch_num'] = b.size()[1]
    model_params['c_dim'] = t.size()[1]
    return model_params

def conduct_cvicdyf_inference(adata, model_params, checkpoint_dirname, epoch=1000, lr=0.003, patience=100, batch_size=128, use_highly_variable=True, two_step=False, dyn_mode=True, module=modules.Cvicdyf, only_vae=False, condition_key='condition', batch_key='sample', cond_in_obsm=False):
    if use_highly_variable:
        adata = adata[:, adata.var.highly_variable]
    # import pdb; pdb.set_trace()
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
    model_params['x_dim'] = s.size()[1]
    model_params['batch_num'] = b.size()[1]
    model_params['c_dim'] = t.size()[1]
    ds = dataset.CvicDyfDataSet(s, u, snorm_mat, unorm_mat, b, t)
    train_ds, val_ds, test_ds = utils.separate_dataset(ds)
    lit_envdyn = optimize_vdicdyf(train_ds, val_ds, model_params, checkpoint_dirname, module, epoch=epoch, lr=lr, patience=patience, batch_size=batch_size, two_step=two_step, dyn_mode=True, only_vae=only_vae)
    envdyn_exp = envdyn.ExpMock(lit_envdyn)
    model = envdyn_exp.model
    adata = commons.update_latent(adata, model, ds)
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata = anndata.concat([
        commons.update_dembed_full(
            adata[idx], model, ds[idx], sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1, fix_embed=True)
        for idx in idx_list], merge='unique')
    # adata = commons.update_dembed_full(adata, model, ds, sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1)
    return adata, lit_envdyn



def conduct_vicdyf_inference(adata, model_params, checkpoint_dirname, epoch=1000, lr=0.003, patience=100, batch_size=128, two_step=False, only_vae=False):
    adata = adata[:, adata.var.highly_variable]
    s, u, snorm_mat, unorm_mat, b, t, adata = make_datasets(adata)
    model_params['x_dim'] = s.size()[1]
    ds = dataset.EnvDynDataSet(s, u, snorm_mat, unorm_mat)
    train_ds, val_ds, test_ds = utils.separate_dataset(ds)
    lit_envdyn = optimize_vdicdyf(train_ds, val_ds, model_params, checkpoint_dirname, modules.EnvDyn, epoch=epoch, lr=lr, patience=patience, batch_size=batch_size, two_step=two_step, dyn_mode=True, only_vae=only_vae)
    envdyn_exp = envdyn.ExpMock(lit_envdyn)
    model = envdyn_exp.model
    # adata = commons.update_dembed_full(adata, model, ds, sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1)
    adata = commons.update_latent(adata, model, ds)
    div_num = math.ceil(adata.shape[0] / 1000)
    idx_list = np.array_split(np.arange(adata.shape[0]), div_num)
    adata = anndata.concat([
        commons.update_dembed_full(
            adata[idx], model, ds[idx], sigma=0.1, embed_mode='vicdyf_', nn=30, mdist=0.1, fix_embed=True)
        for idx in idx_list], merge='unique')
    return adata, lit_envdyn
