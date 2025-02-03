import dataclasses
import datetime
import math
from google.cloud import storage
from typing import Optional
import haiku as hk
from IPython.display import HTML
from IPython import display
import ipywidgets as widgets
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning
import os
import numpy as np
import xarray as xr
import pandas as pd
#####Req function to run model########

#####Loading model#######
# Gives you an authenticated client, in case you want to use a private bucket.
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

params_file_options = [
    name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix+"params/"))
    if (name := blob.name.removeprefix(dir_prefix+"params/"))]  # Drop empty string.

with gcs_bucket.blob(dir_prefix + f"params/GenCast 1p0deg Mini <2019.npz").open("rb") as f:
  ckpt = checkpoint.load(f, gencast.CheckPoint)
  params = ckpt.params
  state = {}

  task_config = ckpt.task_config
  sampler_config = ckpt.sampler_config
  noise_config = ckpt.noise_config
  noise_encoder_config = ckpt.noise_encoder_config

  # Replace attention mechanism.
  splash_spt_cfg = ckpt.denoiser_architecture_config.sparse_transformer_config
  tbd_spt_cfg = dataclasses.replace(splash_spt_cfg, attention_type="triblockdiag_mha", mask_type="full")
  denoiser_architecture_config = dataclasses.replace(ckpt.denoiser_architecture_config, sparse_transformer_config=tbd_spt_cfg)
#   print("Model description:\n", ckpt.description, "\n")
#   print("Model license:\n", ckpt.license, "\n")
  
  
  
  
######LOad Data#######
def loading_data(dataset_file):
    # dataset_file = "era5_new york_2022-01-22.nc"
    data_dir = "./data/"
    # Get the selected dataset file from the dropdown

    # Construct the local file path for the selected file
    local_file_path = os.path.join(data_dir, dataset_file)

    # Check if the file exists locally
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    # Open the selected dataset file from the local file system
    with open(local_file_path, "rb") as f:
        # example_batch = xarray.open_dataset(f, engine="h5netcdf").compute()
        example_batch = xarray.load_dataset(f).compute()
        return example_batch
    # print(example_batch)



def transforming_Data_accor_to_model(example_batch):
    # Latitude and Longitude Arrays
    lat = np.sort(example_batch.coords['lat'])

    lon = example_batch.coords['lon']

    level = example_batch.coords['level']
    time = pd.timedelta_range(start="0h", periods=6, freq="12H")
    datetime = pd.date_range("2019-03-29", periods=6, freq="12H")

    # Create dataset
    example_batch = xr.Dataset(
        {
            "land_sea_mask": ( ["lat", "lon"], np.random.rand(len(lat), len(lon)).astype(np.float32) ),
            "geopotential_at_surface": ( ["lat", "lon"], np.random.randn(len(lat), len(lon)).astype(np.float32) ),
            "day_progress_cos": ( ["batch", "time", "lon"], np.random.randn(1, len(time), len(lon)).astype(np.float32) ),
            "day_progress_sin": ( ["batch", "time", "lon"], np.random.randn(1, len(time), len(lon)).astype(np.float32) ),
            "2m_temperature": ( ["batch", "time", "lat", "lon"], np.random.randn(1, len(time), len(lat), len(lon)).astype(np.float32) ),
            "sea_surface_temperature": ( ["batch", "time", "lat", "lon"], np.random.randn(1, len(time), len(lat), len(lon)).astype(np.float32) ),
            "mean_sea_level_pressure": ( ["batch", "time", "lat", "lon"], np.random.randn(1, len(time), len(lat), len(lon)).astype(np.float32) ),
            "10m_v_component_of_wind": ( ["batch", "time", "lat", "lon"], np.random.randn(1, len(time), len(lat), len(lon)).astype(np.float32) ),
            "total_precipitation_12hr": ( ["batch", "time", "lat", "lon"], np.random.randn(1, len(time), len(lat), len(lon)).astype(np.float32) ),
            "10m_u_component_of_wind": ( ["batch", "time", "lat", "lon"], np.random.randn(1, len(time), len(lat), len(lon)).astype(np.float32) ),
            "u_component_of_wind": ( ["batch", "time", "level", "lat", "lon"], np.random.randn(1, len(time), len(level), len(lat), len(lon)).astype(np.float32) ),
            "specific_humidity": ( ["batch", "time", "level", "lat", "lon"], np.random.randn(1, len(time), len(level), len(lat), len(lon)).astype(np.float32) ),
            "temperature": ( ["batch", "time", "level", "lat", "lon"], np.random.randn(1, len(time), len(level), len(lat), len(lon)).astype(np.float32) ),
            "vertical_velocity": ( ["batch", "time", "level", "lat", "lon"], np.random.randn(1, len(time), len(level), len(lat), len(lon)).astype(np.float32) ),
            "v_component_of_wind": ( ["batch", "time", "level", "lat", "lon"], np.random.randn(1, len(time), len(level), len(lat), len(lon)).astype(np.float32) ),
            "geopotential": ( ["batch", "time", "level", "lat", "lon"], np.random.randn(1, len(time), len(level), len(lat), len(lon)).astype(np.float32) ),
            "year_progress_cos": ( ["batch", "time"], np.random.randn(1, len(time)).astype(np.float32) ),
            "year_progress_sin": ( ["batch", "time"], np.random.randn(1, len(time)).astype(np.float32) ),
        },
        coords={
            "lon": lon,
            "lat": lat,
            "level": level,
            "time": time,
            "datetime": (("batch", "time"), np.array([datetime], dtype="datetime64[ns]")),
        },
    )
    return example_batch
    

def run_model(example_batch):
    # @title Extract training and eval data
    # example_batch = grib_data
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", "12h"), # Only 1AR training.
        **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
        **dataclasses.asdict(task_config))

    print("All Examples:  ", example_batch.dims.mapping)
    print("Train Inputs:  ", train_inputs.dims.mapping)
    print("Train Targets: ", train_targets.dims.mapping)
    print("Train Forcings:", train_forcings.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)
    
    # @title Load normalization data

    with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/min_by_level.nc").open("rb") as f:
        min_by_level = xarray.load_dataset(f).compute()
        
    # @title Build jitted functions, and possibly initialize random weights

    def construct_wrapped_gencast():
        """Constructs and wraps the GenCast Predictor."""
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )

        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )

        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
        )

        return predictor


    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)


    @hk.transform_with_state
    def loss_fn(inputs, targets, forcings):
        predictor = construct_wrapped_gencast()
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )


    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), i, t, f
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True
        )(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads



    loss_fn_jitted = jax.jit(
        lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
    )
    grads_fn_jitted = jax.jit(grads_fn)
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    # We also produce a pmapped version for running in parallel.
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
    # The number of ensemble members should be a multiple of the number of devices.
    print(f"Number of local devices {len(jax.local_devices())}")
    
    # @title Autoregressive rollout (loop in python)
    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    num_ensemble_members = 8 # @param int
    rng = jax.random.PRNGKey(0)

    # We fold-in the ensemble member, this way the first N members should always
    # match across different runs which use take the same inputs, regardless of
    # total ensemble size.
    rngs = np.stack(
        [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)

    chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        # Use pmapped version to parallelise across devices.
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk = 1,
        num_samples = num_ensemble_members,
        pmap_devices=jax.local_devices()
        ):
        chunks.append(chunk)
    predictions = xarray.combine_by_coords(chunks)
    
    print("predictions  ---------->", predictions)
    
    

def model_job():
    data = loading_data('era5_new york_2022-01-32.nc')
    batch = transforming_Data_accor_to_model(data)
    run_model(batch)
    
    
model_job()