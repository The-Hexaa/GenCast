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


# @title Plotting functions

def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xarray.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())


# @title Authenticate with Google Cloud Storage

# Gives you an authenticated client, in case you want to use a private bucket.
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

# @title Choose the model

params_file_options = [
    name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix+"params/"))
    if (name := blob.name.removeprefix(dir_prefix+"params/"))]  # Drop empty string.

latent_value_options = [int(2**i) for i in range(4, 10)]
random_latent_size = widgets.Dropdown(
    options=latent_value_options, value=512,description="Latent size:")
random_attention_type = widgets.Dropdown(
    options=["splash_mha", "triblockdiag_mha", "mha"], value="splash_mha", description="Attention:")
random_mesh_size = widgets.IntSlider(
    value=4, min=4, max=6, description="Mesh size:")
random_num_heads = widgets.Dropdown(
    options=[int(2**i) for i in range(0, 3)], value=4,description="Num heads:")
random_attention_k_hop = widgets.Dropdown(
    options=[int(2**i) for i in range(2, 5)], value=16,description="Attn k hop:")
random_resolution = widgets.Dropdown(
    options=["1p0", "0p25"], value="1p0", description="Resolution:")

def update_latent_options(*args):
  def _latent_valid_for_attn(attn, latent, heads):
    head_dim, rem = divmod(latent, heads)
    if rem != 0:
      return False
    # Required for splash attn.
    if head_dim % 128 != 0:
      return attn != "splash_mha"
    return True
  attn = random_attention_type.value
  heads = random_num_heads.value
  random_latent_size.options = [
      latent for latent in latent_value_options
      if _latent_valid_for_attn(attn, latent, heads)]

# Observe changes to only allow for valid combinations.
random_attention_type.observe(update_latent_options, "value")
random_latent_size.observe(update_latent_options, "value")
random_num_heads.observe(update_latent_options, "value")

params_file = widgets.Dropdown(
    options=[f for f in params_file_options if "Mini" in f],
    description="Params file:",
    layout={"width": "max-content"})

source_tab = widgets.Tab([
    params_file,
    widgets.VBox([
        random_attention_type,
        random_mesh_size,
        random_num_heads,
        random_latent_size,
        random_attention_k_hop,
        random_resolution
    ]),
])
source_tab.set_title(0, "Checkpoint")
source_tab.set_title(1, "Random")
widgets.VBox([
    source_tab,
    widgets.Label(value="Run the next cell to load the model. Rerunning this cell clears your selection.")
])


# @title Load the model

source = source_tab.get_title(source_tab.selected_index)

if source == "Random":
  params = None  # Filled in below
  state = {}
  task_config = gencast.TASK
  # Use default values.
  sampler_config = gencast.SamplerConfig()
  noise_config = gencast.NoiseConfig()
  noise_encoder_config = denoiser.NoiseEncoderConfig()
  # Configure, otherwise use default values.
  denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
    sparse_transformer_config = denoiser.SparseTransformerConfig(
        attention_k_hop=random_attention_k_hop.value,
        attention_type=random_attention_type.value,
        d_model=random_latent_size.value,
        num_heads=random_num_heads.value
        ),
    mesh_size=random_mesh_size.value,
    latent_size=random_latent_size.value,
  )
else:
  assert source == "Checkpoint"
  with gcs_bucket.blob(dir_prefix + f"params/{params_file.value}").open("rb") as f:
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
  print("Model description:\n", ckpt.description, "\n")
  print("Model license:\n", ckpt.license, "\n")

  # @title Get and filter the list of available example datasets

dataset_file_options = [
    name for blob in gcs_bucket.list_blobs(prefix=(dir_prefix + "dataset/"))
    if (name := blob.name.removeprefix(dir_prefix+"dataset/"))]  # Drop empty string.

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))


def data_valid_for_model(file_name: str, params_file_name: str):
  """Check data type and resolution matches."""
  data_file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  data_res = data_file_parts["res"].replace(".", "p")
  if source == "Random":
    return random_resolution.value == data_res
  res_matches = data_res in params_file_name.lower()
  source_matches = "Operational" in params_file_name
  if data_file_parts["source"] == "era5":
    source_matches = not source_matches
  return res_matches and source_matches

dataset_file = widgets.Dropdown(
    options=[
        (", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]), option)
        for option in dataset_file_options
        if data_valid_for_model(option, params_file.value)
    ],
    description="Dataset file:",
    layout={"width": "max-content"})
widgets.VBox([
    dataset_file,
    widgets.Label(value="Run the next cell to load the dataset. Rerunning this cell clears your selection and refilters the datasets that match your model.")
])

# @title Load weather data

with gcs_bucket.blob(dir_prefix+f"dataset/{dataset_file.value}").open("rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.value.removesuffix(".nc")).items()]))

example_batch

# @title Choose data to plot

plot_example_variable = widgets.Dropdown(
    options=example_batch.data_vars.keys(),
    value="2m_temperature",
    description="Variable")
plot_example_level = widgets.Dropdown(
    options=example_batch.coords["level"].values,
    value=500,
    description="Level")
plot_example_robust = widgets.Checkbox(value=True, description="Robust")
plot_example_max_steps = widgets.IntSlider(
    min=1, max=example_batch.dims["time"], value=example_batch.dims["time"],
    description="Max steps")

widgets.VBox([
    plot_example_variable,
    plot_example_level,
    plot_example_robust,
    plot_example_max_steps,
    widgets.Label(value="Run the next cell to plot the data. Rerunning this cell clears your selection.")
])

# @title Plot example data

plot_size = 7

data = {
    " ": scale(select(example_batch, plot_example_variable.value, plot_example_level.value, plot_example_max_steps.value),
              robust=plot_example_robust.value),
}
fig_title = plot_example_variable.value
if "level" in example_batch[plot_example_variable.value].coords:
  fig_title += f" at {plot_example_level.value} hPa"

print(data, fig_title, plot_size, plot_example_robust.value)

# @title Extract training and eval data

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


if params is None:
  init_jitted = jax.jit(loss_fn.init)
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets=train_targets,
      forcings=train_forcings,
  )


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

plot_pred_variable = widgets.Dropdown(
    options=predictions.data_vars.keys(),
    value="2m_temperature",
    description="Variable")
plot_pred_level = widgets.Dropdown(
    options=predictions.coords["level"].values,
    value=500,
    description="Level")
plot_pred_robust = widgets.Checkbox(value=True, description="Robust")
plot_pred_max_steps = widgets.IntSlider(
    min=1,
    max=predictions.dims["time"],
    value=predictions.dims["time"],
    description="Max steps")
plot_pred_samples = widgets.IntSlider(
    min=1,
    max=num_ensemble_members,
    value=num_ensemble_members,
    description="Samples")

widgets.VBox([
    plot_pred_variable,
    plot_pred_level,
    plot_pred_robust,
    plot_pred_max_steps,
    plot_pred_samples,
    widgets.Label(value="Run the next cell to plot the predictions. Rerunning this cell clears your selection.")
])


plot_size = 5
plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)

fig_title = plot_pred_variable.value
if "level" in predictions[plot_pred_variable.value].coords:
  fig_title += f" at {plot_pred_level.value} hPa"

for sample_idx in range(plot_pred_samples.value):
  data = {
      "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
      "Predictions": scale(select(predictions.isel(sample=sample_idx), plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
      "Diff": scale((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps) -
                          select(predictions.isel(sample=sample_idx), plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                        robust=plot_pred_robust.value, center=0),
  }
  display.display(print(data, fig_title + f", Sample {sample_idx}", plot_size, plot_pred_robust.value))


# @title Plot ensemble mean and CRPS

def crps(targets, predictions, bias_corrected = True):
  if predictions.sizes.get("sample", 1) < 2:
    raise ValueError(
        "predictions must have dim 'sample' with size at least 2.")
  sum_dims = ["sample", "sample2"]
  preds2 = predictions.rename({"sample": "sample2"})
  num_samps = predictions.sizes["sample"]
  num_samps2 = (num_samps - 1) if bias_corrected else num_samps
  mean_abs_diff = np.abs(
      predictions - preds2).sum(
          dim=sum_dims, skipna=False) / (num_samps * num_samps2)
  mean_abs_err = np.abs(targets - predictions).sum(dim="sample", skipna=False) / num_samps
  return mean_abs_err - 0.5 * mean_abs_diff


plot_size = 5
plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps.value)

fig_title = plot_pred_variable.value
if "level" in predictions[plot_pred_variable.value].coords:
  fig_title += f" at {plot_pred_level.value} hPa"

data = {
    "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
    "Ensemble Mean": scale(select(predictions.mean(dim=["sample"]), plot_pred_variable.value, plot_pred_level.value, plot_max_steps), robust=plot_pred_robust.value),
    "Ensemble CRPS": scale(crps((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                        select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                      robust=plot_pred_robust.value, center=0),
}
display.display(print(data, fig_title, plot_size, plot_pred_robust.value))

