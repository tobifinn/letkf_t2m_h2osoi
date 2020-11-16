#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 09/23/19
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2019}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
import os
import argparse
import warnings
import glob

# External modules
import xarray as xr
import pandas as pd
from tqdm import tqdm
import distributed

# Internal modules

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
xr.set_options(file_cache_maxsize=1)


def ensure_value(namespace, dest, default):
    stored = getattr(namespace, dest, None)
    if stored is None:
        return value
    return stored


class store_dict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        vals = dict(ensure_value(namespace, self.dest, {}))
        k, _, v = values.partition('=')
        vals[k] = v
        setattr(namespace, self.dest, vals)


parser = argparse.ArgumentParser(
    'Extract variables from given ensemble dataset'
)
parser.add_argument(
    '--var_name', nargs='+', type=str, required=True,
    help='This variable name is extracted from the dataset'
)
parser.add_argument(
    '-p', '--path', type=str, required=True,
    help='This is the path to the data. Regex for different '
    'times are allowed and also the ensemble folder can be '
    'specified with `{0:03d}`.'
)
parser.add_argument(
    '-n', '--n_ens', type=int, required=False, default=40,
    help='Number of ensemble members (default=40)'
)
parser.add_argument(
    '--time_correct', required=False, default=False,
    action='store_true',
    help='If the time axis should be corrected into a normal '
    'datetime axis, this is necessary for CLM '
    '(default=False).'
)
parser.add_argument(
    '-o', '--save_path', type=str, required=True,
    help='This is the output path, where the extracted data is '
    'stored.'
)
parser.add_argument(
    '--isel', default={}, action=store_dict
)
parser.add_argument(
    '--sel', default={}, action=store_dict
)
parser.add_argument(
    '--processes', type=int, required=False, default=1,
    help='Number of parallel processes (default=1)'
)
parser.add_argument(
    '--max_memory', type=float, required=False, default=4096,
    help='System memory in MB. Will be used to define memory per process (default=4096 MB)'
)


def main(args):
    client = init_client(processes=args.processes, max_memory=args.max_memory)
    dataset = load_dataset(path=args.path, ens_mems=args.n_ens)
    var_extracted = extract_var(dataset, args.var_name)
    var_extracted = extract_vcoord(dataset, var_extracted)
    var_extracted = correct_dims(var_extracted)
    var_extracted = sel_coords(var_extracted, args.isel, args.sel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_corrected = correct_time(var_extracted, args.time_correct)
    save_to_nc(var_corrected, args.save_path)


def init_client(processes, max_memory):
    memory_limit = int(max_memory/processes)
    memory_limit = '{0:d}MB'.format(memory_limit)
    logger.info('Initialising client with {0:d} workers and {1:s} per worker'.format(processes, memory_limit))
    cluster = distributed.LocalCluster(n_workers=processes, threads_per_worker=1, memory_limit=memory_limit,
                                       local_directory='/scratch/u/u300636')
    client = distributed.Client(cluster)
    logger.info('Initialised client: {0}'.format(client))
    return client


def load_dataset(path, ens_mems=40):
    logger.info(
        'Extract data from {0:s} for {1:03d} ensemble members'.format(
            path, ens_mems
        )
    )
    ds_ens = []
    pbar_mem = tqdm(range(ens_mems))
    for mem in pbar_mem:
        path_mem = path.format(mem+1)
        pbar_mem.write('Extract {0:s}'.format(path_mem))
        found_paths = sorted(list(glob.glob(path_mem)))
        ds_mem = xr.open_mfdataset(
            found_paths, parallel=True, combine='nested',
            concat_dim='time', chunks={'time': 1}, decode_cf=False,
            decode_times=False
        )
        ds_ens.append(ds_mem)
    logger.info('Starting to concat ensemble')
    ds_ens = xr.concat(ds_ens, dim='ensemble')
    ds_ens = ds_ens.chunk({'ensemble': 1, 'time': 1})
    logger.info('Starting to decode cf-conventions')
    ds_ens = xr.decode_cf(ds_ens)
    logger.info(
        'Concatenated data from {0:s} for {1:03d} ensemble '
        'members'.format(path, ens_mems)
    )
    return ds_ens


def extract_var(ds, var_name):
    try:
        print(var_name)
        da_var = ds[var_name]
        logger.info(
            'Extracted {0} from dataset'.format(var_name)
        )
    except KeyError:
        raise KeyError(
            'Given variable {0:s} is not within the dataset!\n'
            'Available variables: {1:s}'.format(
                var_name, ','.join(list(ds.data_vars.keys()))
            )
        )
    return da_var


def correct_dims(var_extracted):
    missing_keys = [k for k in var_extracted.dims.keys()
                    if k not in var_extracted.coords.keys()]
    for k in missing_keys:
        var_extracted[k] = var_extracted[k]
    return var_extracted


def correct_time(da_var, correct=False):
    if correct:
        da_var = da_var.copy()
        da_var['time'] = da_var.indexes['time'].to_datetimeindex()
        da_var['time'] = da_var.indexes['time'].round('min')
        logger.info(
            'Corrected time dimension'
        )
    return da_var


def sel_coords(da_var, isel, sel):
    isel = {k: eval(v) for k, v in isel.items()}
    sel = {k: eval(v) for k, v in sel.items()}
    da_var = da_var.isel(**isel)
    logger.info('Positional selection of {0}'.format(isel))
    da_var = da_var.sel(**sel)
    logger.info('Selection of {0}'.format(sel))
    return da_var


def extract_vcoord(dataset, var_extracted):
    try:
        vcoord = extract_var(dataset, 'vcoord')
        if isinstance(var_extracted, xr.DataArray):
            var_extracted = var_extracted.to_dataset(name=var_extracted)
        var_extracted['vcoord'] = vcoord
    except KeyError:
        pass
    return var_extracted

def save_to_nc(da_var, save_path):
    logger.info('Save the dataarray to {0:s}'.format(save_path))
    da_var.to_netcdf(save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
