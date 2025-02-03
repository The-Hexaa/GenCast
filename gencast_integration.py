#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to deploy GenCast Mini model, integrate ERA5 historical data,
and incorporate live NOAA Weather API data for a 10-day weather forecast.

Requirements:
  - Python 3.8+
  - JAX, Haiku, NumPy, xarray, requests (for NOAA API), etc.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License:
      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS-IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import requests
import datetime
import math
from typing import Optional, Dict, Any

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import haiku as hk

# Import from graphcast. If you cloned their repository locally, adjust PYTHONPATH or do a local import.
# The user will need to ensure these modules are installed:
#   pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip
# For a real scenario, confirm versions of 'graphcast', 'denoiser', etc.
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

# --------------------------------------------------------------------------------
# Section A: Model + Checkpoint Setup
# --------------------------------------------------------------------------------

def load_gencast_checkpoint(params_file: Optional[str] = None):
    """
    Loads a GenCast checkpoint or initializes random parameters if no file is given.
    The user can customize the path or skip for demonstration.
    """
    if params_file is None:
        print("No checkpoint provided. Initializing GenCast model with random weights.")

        # Build the default config from GenCast
        task_config = gencast.TASK
        sampler_config = gencast.SamplerConfig()
        noise_config = gencast.NoiseConfig()
        noise_encoder_config = denoiser.NoiseEncoderConfig()
        denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
            sparse_transformer_config=denoiser.SparseTransformerConfig(
                attention_k_hop=16,
                attention_type="splash_mha",
                d_model=512,
                num_heads=4,
            ),
            mesh_size=4,
            latent_size=512,
        )
        params = None
        state = {}
        checkpoint_description = "Randomly initialized GenCast Model"
        checkpoint_license = "Apache 2.0 (example demonstration)"
    else:
        # Use a real checkpoint
        print(f"Loading checkpoint from: {params_file}")
        with open(params_file, "rb") as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)
        params = ckpt.params
        state = {}
        task_config = ckpt.task_config
        sampler_config = ckpt.sampler_config
        noise_config = ckpt.noise_config
        noise_encoder_config = ckpt.noise_encoder_config
        denoiser_architecture_config = ckpt.denoiser_architecture_config
        checkpoint_description = ckpt.description
        checkpoint_license = ckpt.license

    return (params, state,
            task_config,
            sampler_config,
            noise_config,
            noise_encoder_config,
            denoiser_architecture_config,
            checkpoint_description,
            checkpoint_license)


def construct_gencast_model(task_config,
                            sampler_config,
                            noise_config,
                            noise_encoder_config,
                            denoiser_architecture_config,
                            diffs_stddev_by_level,
                            mean_by_level,
                            stddev_by_level,
                            min_by_level):
    """
    Constructs the GenCast predictor pipeline with transformations, normalization,
    and NaN cleaning.
    """
    def build_predictor():
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
        # Example: Clean sea surface temperature
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
        )
        return predictor

    # Transform forward function
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = build_predictor()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    return run_forward


# --------------------------------------------------------------------------------
# Section B: Loading Historical ERA5 Data
# --------------------------------------------------------------------------------

def load_era5_data(era5_path: str) -> xr.Dataset:
    """
    Loads historical ERA5 data from a NetCDF or Zarr file.
    Adjust path/method as needed. Must be consistent with model resolution.
    """
    print(f"Loading ERA5 data from: {era5_path}")
    era5_ds = xr.open_dataset(era5_path)
    return era5_ds


# --------------------------------------------------------------------------------
# Section C: Integrating Live NOAA Data
# --------------------------------------------------------------------------------

def fetch_noaa_live_data(city: str) -> Dict[str, Any]:
    """
    Example placeholder function to fetch real-time data from NOAA's API for a given city.
    NOAA's public endpoints often require lat/lon or station ID, so we do:
      - find lat/lon from city name with some geocoding service
      - call NOAA's endpoint
      - parse JSON
    Here, we simply return a mocked dictionary.
    """
    print(f"Fetching live NOAA data for {city} ... (placeholder)")

    # Example (NOT real NOAA endpoint):
    #   station_id = "GHCND:USW00094728"  # e.g., for Chicago O'Hare
    #   response = requests.get(f"https://api.weather.gov/stations/{station_id}/observations/latest")
    #   data = response.json()

    # For demonstration, let's pretend the NOAA data returned is a dictionary:
    live_data_example = {
        "2m_temperature": 295.15,     # 22 °C
        "surface_pressure": 101300.0, # Pa
        "sea_surface_temperature": 299.15,
        # ... additional data
    }
    return live_data_example


def merge_era5_and_noaa(era5_ds: xr.Dataset, noaa_data: Dict[str, Any]) -> xr.Dataset:
    """
    Merges the ERA5 dataset with the latest NOAA data as initial condition.
    Typically, you'd have to carefully align variables, dims, etc.
    For simplicity, we inject NOAA values into the last time step of ERA5.
    """
    # In a real solution, you'd align lat/lon, pressure levels, etc.
    # We'll assume 2m_temperature and other variables exist in era5_ds.
    # Example: override the final time step with NOAA data
    last_time_index = era5_ds["time"][-1]

    # We'll create a copy with updated final time step values
    era5_updated = era5_ds.copy()

    for var, val in noaa_data.items():
        if var in era5_updated.data_vars:
            # This is simplistic. Real logic requires regridding or spatial matching.
            # We demonstrate the concept by setting the entire final slice to NOAA value.
            era5_updated[var].loc[{"time": last_time_index}] = val

    return era5_updated


# --------------------------------------------------------------------------------
# Section D: Model Inference & 10-Day Forecast
# --------------------------------------------------------------------------------

def run_10day_forecast(run_forward,
                       params,
                       state,
                       era5_noaa_ds: xr.Dataset,
                       diffs_stddev_by_level,
                       mean_by_level,
                       stddev_by_level,
                       min_by_level,
                       lead_hours: int = 240,  # 10 days * 24 hours
                       num_ensemble_members: int = 4) -> xr.Dataset:
    """
    Runs an autoregressive forecast for up to 'lead_hours' into the future.
    Returns an xarray Dataset of predictions.
    """

    # Example: We create "inputs" and "forcings" from the dataset
    # We'll do a minimal version of data_utils.extract_inputs_targets_forcings
    # to form a single "batch" example. For real usage, you'd adapt more thoroughly.
    # We assume the last 2 time-steps are inputs, the rest are for evaluating model outputs.
    # This is an overly simplified approach to demonstrate.
    time_dim = era5_noaa_ds.dims["time"]
    if time_dim < 3:
        raise ValueError("Not enough time steps in ERA5 + NOAA data for model inputs!")

    # Example approach: use the last 2 time steps as "inputs"
    # and keep placeholders for the next 10 days (the model will roll them out).
    # The GenCast code typically uses data_utils, so let's just do an example:
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        era5_noaa_ds,
        target_lead_times=slice("12h", "12h"),  # minimal placeholder
        # input_time_chunk_size=2,
        # overlap_time_chunk_size=1
    )

    # We can define an autoregressive rollout, similar to the chunked method from the original:
    def forward_jitted(rng, i, t, f):
        return run_forward.apply(params, state, rng, i, t, f)[0]

    run_forward_pmap = xarray_jax.pmap(jax.jit(forward_jitted), dim="sample")

    # Generate random seeds for ensemble
    rng_key = jax.random.PRNGKey(42)
    rngs = np.stack(
        [jax.random.fold_in(rng_key, i) for i in range(num_ensemble_members)],
        axis=0
    )

    # Use chunked_prediction_generator_multiple_runs to iterate in an autoregressive manner:
    chunked_preds = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
        num_steps_per_chunk=1,
        num_samples=num_ensemble_members,
        pmap_devices=jax.local_devices()
    ):
        chunked_preds.append(chunk)

    predictions = xr.combine_by_coords(chunked_preds)
    return predictions


# --------------------------------------------------------------------------------
# Section E: Main Function
# --------------------------------------------------------------------------------

def main():
    """
    1. Load or initialize GenCast model parameters.
    2. Fetch ERA5 historical data and NOAA live data, merge them.
    3. Run GenCast model for next 10 days (hourly or daily) predictions.
    4. Save or print out results.
    """

    # 1) Load the checkpoint or random init
    # Provide a path if you have a real checkpoint, else keep None for random.
    params_file = None  # e.g. "my_gencast_mini_checkpoint.ckpt"
    (params, state,
     task_config,
     sampler_config,
     noise_config,
     noise_encoder_config,
     denoiser_architecture_config,
     checkpoint_description,
     checkpoint_license) = load_gencast_checkpoint(params_file)

    print("Checkpoint description:", checkpoint_description)
    print("Checkpoint license:", checkpoint_license)

    # 2) Load normalization stats (ERA5 based) if you have them.
    # For demonstration, we create placeholders or load from local files:
    # These are typically NetCDF or xarray files: diffs_stddev_by_level.nc, etc.
    # For real usage, uncomment and replace with your file paths.
    #   diffs_stddev_by_level = xr.open_dataset("diffs_stddev_by_level.nc")
    #   mean_by_level = xr.open_dataset("mean_by_level.nc")
    #   stddev_by_level = xr.open_dataset("stddev_by_level.nc")
    #   min_by_level = xr.open_dataset("min_by_level.nc")
    # Instead, let's do minimal placeholders for demonstration:
    diffs_stddev_by_level = xr.Dataset()
    mean_by_level = xr.Dataset()
    stddev_by_level = xr.Dataset()
    min_by_level = xr.Dataset()

    # 3) Construct the model
    run_forward = construct_gencast_model(
        task_config,
        sampler_config,
        noise_config,
        noise_encoder_config,
        denoiser_architecture_config,
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
        min_by_level
    )

    # If we used random initialization, we must init:
    if params is None:
        print("Initializing random parameters for GenCast model.")
        # We create a dummy dataset just for initialization shape
        dummy_ds = xr.Dataset(
            {
                "2m_temperature": (("time", "lat", "lon"), np.random.rand(3, 10, 10)),
            },
            coords={
                "time": np.array([0, 1, 2]),
                "lat": np.linspace(-90, 90, 10),
                "lon": np.linspace(-180, 180, 10),
            },
        )
        dummy_inputs, dummy_targets, dummy_forcings = data_utils.extract_inputs_targets_forcings(
            dummy_ds,
            target_lead_times=slice("12h", "12h"),
            # input_time_chunk_size=2,
            # overlap_time_chunk_size=1
        )
        init_fn = jax.jit(run_forward.init)
        params, st = init_fn(
            jax.random.PRNGKey(0),
            dummy_inputs,
            dummy_targets,
            dummy_forcings
        )
        # We'll keep the model state as empty or st from init
        state = st

    # 4) Load ERA5 data
    era5_path = "example_era5.nc"  # Replace with your actual file
    if not os.path.exists(era5_path):
        print(f"Warning: '{era5_path}' not found. Using a dummy dataset.")
        era5_ds = xr.Dataset(
            {
                "2m_temperature": (("time", "lat", "lon"), np.random.rand(5, 10, 10)),
                "surface_pressure": (("time", "lat", "lon"), 100000 + 500*np.random.rand(5,10,10)),
                "sea_surface_temperature": (("time", "lat", "lon"), 290 + 5*np.random.rand(5,10,10)),
            },
            coords={
                "time": np.array([0, 1, 2, 3, 4]),
                "lat": np.linspace(-90, 90, 10),
                "lon": np.linspace(-180, 180, 10),
            },
        )
    else:
        era5_ds = load_era5_data(era5_path)

    # 5) Fetch NOAA data for a city (placeholder example).
    city = "Chicago"
    noaa_data = fetch_noaa_live_data(city)

    # 6) Merge the NOAA data with ERA5's final time slice (naive approach).
    merged_ds = merge_era5_and_noaa(era5_ds, noaa_data)

    # 7) Run 10-day forecast:
    predictions = run_10day_forecast(
        run_forward,
        params,
        state,
        merged_ds,
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
        min_by_level,
        lead_hours=240,  # 10 days * 24 hours
        num_ensemble_members=4
    )

    print("Forecast complete. Sample of predictions data structure:")
    print(predictions)

    # 8) Save or handle final output
    # For instance, predictions.to_netcdf("10_day_forecast_output.nc")
    # or any other usage (plot, etc.).
    print("Done.")


if __name__ == "__main__":
    main()
