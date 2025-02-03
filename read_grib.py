# import os
# import xarray as xr
# import numpy as np

# # Define the data directory
# data_dir = "data"

# # Find the first .grib file in the data directory
# grib_files = [f for f in os.listdir(data_dir) if f.endswith('era5_data.grib')]

# if not grib_files:
#     print("No .grib file found in the 'data' directory.")
#     exit()

# grib_file = os.path.join(data_dir, grib_files[0])
# print(f"Reading {grib_file}...")

# # Load GRIB file
# ds = xr.open_dataset(grib_file, engine="cfgrib")

# # Convert longitude from [-180, 180] to [0, 360]
# ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")

# # Rename coordinates to match expected format
# ds = ds.rename({"longitude": "lon", "latitude": "lat", "valid_time": "datetime"})

# # Add missing dimensions if they don't exist
# if "time" not in ds.dims:
#     # Set time to timedelta64[ns] format, starting from zero
#     time_values = np.array([np.timedelta64(0, 'm'), np.timedelta64(24, 'h'), np.timedelta64(48, 'h')])
#     ds.coords['time'] = ('time', time_values)

# if "level" not in ds.dims:
#     levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 925, 1000])
#     ds = ds.expand_dims(level=levels)

# # Expand batch dimension
# ds = ds.expand_dims(batch=1)

# # Rename data variables to match the expected names
# var_rename = {
#     "lsm": "land_sea_mask",
#     "z": "geopotential_at_surface",
#     "sst": "sea_surface_temperature",
#     "msl": "mean_sea_level_pressure",
#     "v10": "v_component_of_wind",
#     "u10": "u_component_of_wind"
# }
# ds = ds.rename(var_rename)

# # Remove unnecessary coordinates (keep only lat, lon, level, time, and datetime)
# coords_to_keep = ['lat', 'lon', 'level', 'time', 'datetime', 'batch']
# coords_to_remove = [coord for coord in ds.coords if coord not in coords_to_keep]

# # Use drop_vars to remove the coordinates
# ds = ds.drop_vars(coords_to_remove)

# # Ensure 'lon' has the correct length (1440)
# lon_size = 1440  # Set to the correct size from your data
# ds['day_progress_cos'] = (('batch', 'time', 'lon'), np.ones((1, 3, lon_size), dtype=np.float32))  # Example dummy data
# ds['day_progress_sin'] = (('batch', 'time', 'lon'), np.zeros((1, 3, lon_size), dtype=np.float32))  # Example dummy data

# # Save cleaned dataset to NetCDF
# output_file = "data/transformed_data.nc"
# ds.to_netcdf(output_file)
# print(f"Cleaned data saved to {output_file}")

# # Display dataset after processing
# print(ds)






import os
import xarray as xr
import numpy as np

# Define the data directory
data_dir = "data"

# Find the first .grib file in the data directory
grib_files = [f for f in os.listdir(data_dir) if f.endswith('.grib')]

if not grib_files:
    print("No .grib file found in the 'data' directory.")
    exit()

grib_file = os.path.join(data_dir, grib_files[0])
print(f"Reading {grib_file}...")

# Load GRIB file
ds = xr.open_dataset(grib_file, engine="cfgrib")

# Convert longitude from [-180, 180] to [0, 360]
ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")

# Rename coordinates to match expected format
ds = ds.rename({"longitude": "lon", "latitude": "lat", "valid_time": "datetime"})

# Add missing dimensions if they don't exist
if "time" not in ds.dims:
    ds = ds.expand_dims(time=[0])

if "level" not in ds.dims:
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 925, 1000])
    ds = ds.expand_dims(level=levels)

# Rename data variables to match the expected names
var_rename = {
    "lsm": "land_sea_mask",
    "z": "geopotential_at_surface",
    "sst": "sea_surface_temperature",
    "msl": "mean_sea_level_pressure",
    "v10": "v_component_of_wind",
    "u10": "u_component_of_wind"
}
ds = ds.rename(var_rename)

# Remove unnecessary coordinates (keep only lat, lon, level, time, and datetime)
coords_to_keep = ['lat', 'lon', 'level', 'time', 'datetime']
coords_to_remove = [coord for coord in ds.coords if coord not in coords_to_keep]

# Use drop method to remove the coordinates
ds = ds.drop(coords_to_remove)

# Save cleaned dataset to NetCDF
output_file = "data/transformed_data.nc"
ds.to_netcdf(output_file)
print(f"Cleaned data saved to {output_file}")

# Display dataset after processing
print(ds)
