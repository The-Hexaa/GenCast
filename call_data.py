import os
import stat
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import numpy.ma as ma
from netCDF4 import Dataset, date2num
from flask import Flask, request, jsonify, render_template, url_for

# Configure logger
logger = logging.getLogger(__name__)

# Configuration
NOAA_API_TOKEN = "qfgAteWLTqygunRnlVLnZaTAwbOXBrlT"
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

def get_location_id_by_city(city_name):
    city_location_map = {
        "New York": "FIPS:36",
        "Los Angeles": "FIPS:06",
        "Chicago": "FIPS:17",
    }
    return city_location_map.get(city_name)

def fetch_noaa_data(dataset_id, start_date, end_date, location_id, parameters):
    """Enhanced NOAA data fetching with better error handling and data validation"""
    headers = {"token": NOAA_API_TOKEN}
    all_data = []
    offset = 0
    limit = 1000
    
    for data_type in parameters:
        while True:
            params = {
                "datasetid": dataset_id,
                "startdate": start_date,
                "enddate": end_date,
                "locationid": location_id,
                "datatypeid": data_type,
                "limit": limit,
                "offset": offset,
                "units": "metric"
            }
            
            try:
                response = requests.get(f"{BASE_URL}data", headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("results"):
                    break
                    
                all_data.extend(data["results"])
                
                # Check if we've got all the data
                if len(data["results"]) < limit:
                    break
                    
                offset += limit
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching {data_type}: {str(e)}")
                break
    
    return {"results": all_data}

def get_dimensions():
    # Fixed dimensions matching ERA5 format
    longitudes = np.linspace(0, 359, 360)
    latitudes = np.linspace(-90, 90, 181)
    levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    return latitudes, longitudes, levels

def transform_to_era5_format(noaa_data, latitudes, longitudes, levels, time_steps):
    """Transform NOAA data to match ERA5 format exactly"""
    num_times = len(time_steps)
    
    # Initialize data structure matching ERA5 format exactly
    data = {
        # 4D variables (batch, time, level, lat, lon)
        "temperature": ma.masked_all((1, num_times, len(levels), len(latitudes), len(longitudes)), dtype=np.float32),
        "vertical_velocity": ma.masked_all((1, num_times, len(levels), len(latitudes), len(longitudes)), dtype=np.float32),
        "v_component_of_wind": ma.masked_all((1, num_times, len(levels), len(latitudes), len(longitudes)), dtype=np.float32),
        "geopotential": ma.masked_all((1, num_times, len(levels), len(latitudes), len(longitudes)), dtype=np.float32),
        
        # 3D variables (batch, time, lat, lon)
        "2m_temperature": ma.masked_all((1, num_times, len(latitudes), len(longitudes)), dtype=np.float32),
        "sea_surface_temperature": ma.masked_all((1, num_times, len(latitudes), len(longitudes)), dtype=np.float32),
        
        # 2D variables (lat, lon)
        "land_sea_mask": ma.masked_all((len(latitudes), len(longitudes)), dtype=np.float32),
        "geopotential_at_surface": ma.masked_all((len(latitudes), len(longitudes)), dtype=np.float32),
        
        # Time-based variables
        "day_progress_cos": ma.masked_all((1, num_times, len(longitudes)), dtype=np.float32),
        "day_progress_sin": ma.masked_all((1, num_times, len(longitudes)), dtype=np.float32),
        "year_progress_cos": ma.masked_all((1, num_times), dtype=np.float32),
        "year_progress_sin": ma.masked_all((1, num_times), dtype=np.float32)
    }
    
    # Fill time-based variables
    for t, timestamp in enumerate(time_steps):
        hour = timestamp.hour + timestamp.minute/60
        day_progress = 2 * np.pi * hour / 24
        data["day_progress_cos"][0, t, :] = np.cos(day_progress)
        data["day_progress_sin"][0, t, :] = np.sin(day_progress)
        
        doy = timestamp.timetuple().tm_yday
        year_progress = 2 * np.pi * doy / 365.25
        data["year_progress_cos"][0, t] = np.cos(year_progress)
        data["year_progress_sin"][0, t] = np.sin(year_progress)
    
    # Create a time index map
    time_map = {ts.strftime("%Y-%m-%d"): idx for idx, ts in enumerate(time_steps)}
    
    # Process NOAA data
    for result in noaa_data.get("results", []):
        try:
            # Parse NOAA data
            date_str = result.get("date", "").split("T")[0]
            if date_str not in time_map:
                continue
                
            time_idx = time_map[date_str]
            lat = float(result.get("latitude", 0))
            lon = float(result.get("longitude", 0))
            value = float(result.get("value", 0))
            
            # Find nearest grid points
            lat_idx = np.abs(latitudes - lat).argmin()
            lon_idx = np.abs(longitudes - lon).argmin()
            
            # Process different data types
            data_type = result.get("datatype")
            if data_type == "TEMP":
                # Convert Celsius to Kelvin
                temp_k = value + 273.15
                data["2m_temperature"][0, time_idx, lat_idx, lon_idx] = temp_k
                
                # Estimate temperature profile
                for l, level in enumerate(levels):
                    pressure_ratio = level / 1000.0  # normalize by surface pressure
                    lapse_rate = 0.0065  # standard atmosphere lapse rate K/m
                    height = -7000 * np.log(pressure_ratio)  # approximate height in meters
                    temp_at_level = temp_k - (lapse_rate * height)
                    data["temperature"][0, time_idx, l, lat_idx, lon_idx] = temp_at_level
                    
            elif data_type == "PRCP":
                # Convert mm to m
                precip_m = value / 1000.0
                data["precipitation"][0, time_idx, lat_idx, lon_idx] = precip_m
                
            elif data_type == "WIND":
                # Decompose wind speed into u and v components
                # Assuming wind direction is available, otherwise use climatological values
                wind_direction = float(result.get("direction", 225))  # default to southwest
                wind_rad = np.radians(wind_direction)
                
                u_wind = -value * np.sin(wind_rad)
                v_wind = -value * np.cos(wind_rad)
                
                for l, _ in enumerate(levels):
                    # Apply simple wind profile
                    height_factor = np.sqrt(1000/levels[l])
                    data["u_component_of_wind"][0, time_idx, l, lat_idx, lon_idx] = u_wind * height_factor
                    data["v_component_of_wind"][0, time_idx, l, lat_idx, lon_idx] = v_wind * height_factor
            
            elif data_type == "PRES":
                # Convert hPa to Pa for geopotential calculation
                pressure_pa = value * 100
                for l, level in enumerate(levels):
                    height = 7000 * np.log(1000/level)  # approximate height in meters
                    data["geopotential"][0, time_idx, l, lat_idx, lon_idx] = 9.81 * height
                data["geopotential_at_surface"][lat_idx, lon_idx] = height * 9.81
            
            # Set land-sea mask (1 for land, 0 for sea)
            data["land_sea_mask"][lat_idx, lon_idx] = 1.0
                    
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error processing record: {str(e)}")
            continue
            
    return data

def ensure_directory_permissions():
    """Ensure output directory exists with proper permissions"""
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Set directory permissions to allow writing
        os.chmod(OUTPUT_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except Exception as e:
        raise RuntimeError(f"Failed to set up output directory: {str(e)}")

def save_to_netcdf(era5_data, latitudes, longitudes, levels, time_steps, output_file):
    """Save data matching ERA5 format exactly"""
    ensure_directory_permissions()
    
    nc_file_path = os.path.join(OUTPUT_DIR, output_file)
    with Dataset(nc_file_path, "w", format="NETCDF4") as nc_file:
        # Create dimensions matching ERA5 exactly
        nc_file.createDimension("lon", 360)
        nc_file.createDimension("lat", 181)
        nc_file.createDimension("level", 13)
        nc_file.createDimension("time", len(time_steps))
        nc_file.createDimension("batch", 1)

        # Create coordinate variables with exact specifications
        coordinates = {
            "lon": {
                "dims": ("lon",),
                "data": np.linspace(0, 359, 360),
                "dtype": "f4",
                "attrs": {"long_name": "longitude", "units": "degrees_east"}
            },
            "lat": {
                "dims": ("lat",),
                "data": np.linspace(-90, 90, 181),
                "dtype": "f4",
                "attrs": {"long_name": "latitude", "units": "degrees_north"}
            },
            "level": {
                "dims": ("level",),
                "data": np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]),
                "dtype": "i4",
                "attrs": {"long_name": "pressure_level", "units": "hPa"}
            },
            "time": {
                "dims": ("time",),
                "dtype": "timedelta64[ns]",
                "attrs": {"long_name": "time"}
            },
            "datetime": {
                "dims": ("batch", "time"),
                "dtype": "datetime64[ns]",
                "attrs": {"long_name": "datetime"}
            }
        }

        # Create and fill coordinate variables
        for name, config in coordinates.items():
            var = nc_file.createVariable(name, config["dtype"], config["dims"])
            for key, value in config.get("attrs", {}).items():
                setattr(var, key, value)
            if "data" in config:
                var[:] = config["data"]

        # Set time coordinates
        time_var = nc_file.variables["time"]
        time_deltas = np.array([np.timedelta64(i, 'D') for i in range(len(time_steps))])
        time_var[:] = time_deltas

        datetime_var = nc_file.variables["datetime"]
        datetime_values = np.array(time_steps).astype('datetime64[ns]').reshape(1, -1)
        datetime_var[:] = datetime_values

        # Create data variables with exact specifications
        variables = {
            "land_sea_mask": {
                "dims": ("lat", "lon"),
                "dtype": "f4",
                "size": "261kB",
                "attrs": {"long_name": "Land-sea mask"}
            },
            "geopotential_at_surface": {
                "dims": ("lat", "lon"),
                "dtype": "f4",
                "size": "261kB",
                "attrs": {"long_name": "Geopotential at surface"}
            },
            "day_progress_cos": {
                "dims": ("batch", "time", "lon"),
                "dtype": "f4",
                "size": "4kB",
                "attrs": {"long_name": "Cosine of day progress"}
            },
            "day_progress_sin": {
                "dims": ("batch", "time", "lon"),
                "dtype": "f4",
                "size": "4kB",
                "attrs": {"long_name": "Sine of day progress"}
            },
            "2m_temperature": {
                "dims": ("batch", "time", "lat", "lon"),
                "dtype": "f4",
                "size": "782kB",
                "attrs": {"long_name": "2 metre temperature", "units": "K"}
            },
            "sea_surface_temperature": {
                "dims": ("batch", "time", "lat", "lon"),
                "dtype": "f4",
                "size": "782kB",
                "attrs": {"long_name": "Sea surface temperature", "units": "K"}
            },
            "temperature": {
                "dims": ("batch", "time", "level", "lat", "lon"),
                "dtype": "f4",
                "size": "10MB",
                "attrs": {"long_name": "Temperature", "units": "K"}
            },
            "vertical_velocity": {
                "dims": ("batch", "time", "level", "lat", "lon"),
                "dtype": "f4",
                "size": "10MB",
                "attrs": {"long_name": "Vertical velocity", "units": "Pa/s"}
            },
            "v_component_of_wind": {
                "dims": ("batch", "time", "level", "lat", "lon"),
                "dtype": "f4",
                "size": "10MB",
                "attrs": {"long_name": "V component of wind", "units": "m/s"}
            },
            "geopotential": {
                "dims": ("batch", "time", "level", "lat", "lon"),
                "dtype": "f4",
                "size": "10MB",
                "attrs": {"long_name": "Geopotential", "units": "m^2/s^2"}
            },
            "year_progress_cos": {
                "dims": ("batch", "time"),
                "dtype": "f4",
                "size": "12B",
                "attrs": {"long_name": "Cosine of year progress"}
            },
            "year_progress_sin": {
                "dims": ("batch", "time"),
                "dtype": "f4",
                "size": "12B",
                "attrs": {"long_name": "Sine of year progress"}
            }
        }

        # Create and fill data variables
        for var_name, config in variables.items():
            if var_name in era5_data:
                var = nc_file.createVariable(
                    var_name,
                    config["dtype"],
                    config["dims"],
                    zlib=True,
                    complevel=4,
                    shuffle=True
                )
                for key, value in config["attrs"].items():
                    setattr(var, key, value)
                var[:] = era5_data[var_name]

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/post_data", methods=["POST"])
def post_data():
    try:
        ensure_directory_permissions()
        data = request.json
        if not data:
            raise ValueError("No data provided")
        
        required_fields = ["dataset_id", "start_date", "end_date", "city_name", "parameters"]
        if not all(field in data for field in required_fields):
            raise ValueError(f"Missing required fields. Required: {required_fields}")
        
        location_id = get_location_id_by_city(data["city_name"])
        if not location_id:
            raise ValueError(f"Invalid city name: {data['city_name']}")
        
        # Get dimensions and fetch data
        latitudes, longitudes, levels = get_dimensions()
        time_steps = pd.date_range(start=data["start_date"], end=data["end_date"], freq='D').tolist()
        
        # Fetch and transform data
        noaa_data = fetch_noaa_data(
            data["dataset_id"],
            data["start_date"],
            data["end_date"],
            location_id,
            data["parameters"]
        )
        
        era5_data = transform_to_era5_format(
            noaa_data,
            latitudes,
            longitudes,
            levels,
            time_steps
        )
        
        # Save to NetCDF
        output_filename = f"era5_{data['city_name'].lower()}_{data['start_date']}_to_{data['end_date']}.nc"
        save_to_netcdf(
            era5_data,
            latitudes,
            longitudes,
            levels,
            time_steps,
            output_filename
        )
        
        return jsonify({
            "status": "success",
            "message": f"Data saved to {output_filename}",
            "file_path": os.path.join(OUTPUT_DIR, output_filename),
            "dimensions": {
                "lon": len(longitudes),
                "lat": len(latitudes),
                "level": len(levels),
                "time": len(time_steps),
                "batch": 1
            }
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)