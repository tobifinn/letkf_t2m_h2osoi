# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: pytassim
#     language: python
#     name: pytassim
# ---

# +
import argparse
import pickle as pk
import logging
import os
import glob
from functools import partial

import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
import distributed
from dask import delayed
from dask.diagnostics import ProgressBar

import pytassim
from pytassim.assimilation import LETKFUncorr
from pytassim.assimilation.filter.letkf import local_etkf
from pytassim.localization import GaspariCohn
from pytassim.obs_ops.terrsysmp.cos_t2m import CosmoT2mOperator, EARTH_RADIUS
from pytassim.model.terrsysmp import preprocess_cosmo, postprocess_cosmo


# Static settings
VR_BASE_PATH = "/p/scratch/chbn29/hbn29p/data/tsmp/runs/da_enkf_for_soil/016/t2m_cleaned.nc"
OBS_PATH = "/p/scratch/chbn29/hbn29p/data/tsmp/runs/obs/ens/t2m_obs_016_0_1_long.nc"
STATIONS_PATH = '/p/scratch/chbn29/hbn29p/data/tsmp/runs/utilities/stations.hd5'
CONST_PATH = '/p/scratch/chbn29/hbn29p/data/tsmp/runs/utilities/cosmo_const.nc'
COORDS_PATH ='/p/scratch/chbn29/hbn29p/data/tsmp/runs/utilities/cosmo_grid.pk'


# Class initializations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
rnd = np.random.RandomState(42)


def load_vr_ds(path):
    loaded_ds = xr.open_dataset(path)
    return loaded_ds


def load_vr(file_path, const_path, coords_path):
    vr_cos_ds = load_vr_ds(file_path)
    vr_cos_const = xr.open_dataset(const_path).load()
    vr_cos_coords = coords_latlon = np.stack(
        pk.load(open(coords_path, 'rb'))._calc_lat_lon(), axis=-1
    )
    return vr_cos_ds, vr_cos_const, vr_cos_coords


def generate_observations(cosmo_ds, obs_op, obs_bias=0, obs_stddev=0.2):
    raw_observations = obs_op.obs_op(cosmo_ds)
    obs_pert = rnd.normal(loc=obs_bias, scale=obs_stddev, size=raw_observations.shape)
    observations = raw_observations + obs_pert
    observations = observations.transpose('time', 'grid')
    observations = observations.expand_dims('variable', axis=0)
    observations = observations.isel(variable=0).rename(grid='obs_grid_1')

    obs_cov = xr.DataArray(
        [obs_stddev**2] * len(observations.obs_grid_1),
        coords={
            'obs_grid_1': observations.obs_grid_1.values,
        },
        dims=['obs_grid_1',]
    )
    obs_ds = xr.Dataset({'observations': observations, 'covariance': obs_cov})
    return obs_ds


def main(src_base_path, trg_file, const_path, coords_path, stations_path, obs_bias=0, obs_stddev=0.2):
    logger.info('Starting with analysis script')
    vr_file_path = src_base_path
    vr_cos_ds, vr_const_cos, vr_cos_coords = load_vr(vr_file_path, const_path, coords_path)
    logger.info('Finished loading VR data')

    _, index = np.unique(vr_cos_ds['time'], return_index=True)
    vr_cos_ds = vr_cos_ds.isel(time=index)
    if not 'vcoord' in vr_cos_ds.data_vars.keys():
        vr_cos_ds['vcoord'] = vr_const_cos['vcoord']

    vr_cos_data = preprocess_cosmo(vr_cos_ds, ['T_2M'])

    vr_cos_data = vr_cos_data.load().squeeze('ensemble')
    station_df = pd.read_hdf(stations_path, 'stations')
    obs_operator = CosmoT2mOperator(station_df, cosmo_coords=vr_cos_coords,
                                    cosmo_const=vr_const_cos)

    # ### Lapse rate to fixed lapse rate of 0.7 K / 100 m
    obs_operator.get_lapse_rate = lambda x: -0.007

    observations = generate_observations(vr_cos_data, obs_operator, obs_bias, obs_stddev)
    logger.info('Finished generating observations')

    obs_flatten = observations.reset_index('obs_grid_1')
    obs_flatten.attrs['multiindex']=list(observations.indexes['obs_grid_1'].names)
    obs_flatten.to_netcdf(trg_file)
    logger.info('Saved observations to {0:s}'.format(trg_file))



if __name__ == '__main__':
    #client = distributed.get_client(CLIENT_ADDRESS)
    main(VR_BASE_PATH, OBS_PATH, CONST_PATH, COORDS_PATH,
         STATIONS_PATH, obs_bias=0, obs_stddev=0.1)
