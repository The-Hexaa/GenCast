from flask import Flask, request, jsonify
import requests
from netCDF4 import Dataset
import numpy as np
import os
import urllib.request
import urllib.parse
import json

NOAA_API_TOKEN = "qfgAteWLTqygunRnlVLnZaTAwbOXBrlT"
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
OUTPUT_DIR = "./data"  # Directory to save NetCDF files

app = Flask(__name__)

# Generate latitude and longitude arrays
lat_array = np.arange(-90, 90.25, 0.25, dtype=np.float32)  # -90 to 90 with 0.25 step
lon_array = np.arange(0, 360.25, 5, dtype=np.float32)  # 0 to 360 with 5 step

# Function to get latitude and longitude using OpenStreetMap Nominatim API
def get_lat_lon_from_city(city_name):
    url = "http://nominatim.openstreetmap.org/search?"
    params = urllib.parse.urlencode({"q": city_name, "format": "json"})

    try:
        response = urllib.request.urlopen(url + params)
        data = json.loads(response.read().decode("utf-8"))
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            
            # Ensure longitude is in 0-360 range
            if lon < 0:
                lon += 360

            # Get the closest available latitude and longitude
            lat = lat_array[np.abs(lat_array - lat).argmin()]
            lon = lon_array[np.abs(lon_array - lon).argmin()]
            
            return lat, lon
    except Exception as e:
        print(f"Error fetching lat/lon: {e}")
    return None, None  # Return None if coordinates couldn't be found

# Function to fetch data from NOAA API
def fetch_noaa_data(dataset_id, date, location_id, data_type):
    headers = {"token": NOAA_API_TOKEN}
    params = {
        "datasetid": dataset_id,
        "startdate": date,
        "enddate": date,
        "locationid": location_id,
        "datatypeid": data_type,
        "limit": 1,
    }

    response = requests.get(BASE_URL + "data", headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Error fetching data: {response.status_code}, {response.text}")

# Function to transform NOAA data and include lat/lon
def transform_to_era5_format(noaa_data, city_name):
    lat, lon = get_lat_lon_from_city(city_name)  # Fetch coordinates

    features = []
    for result in noaa_data.get("results", []):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],  # Use fetched coordinates
            },
            "properties": {
                "date": result["date"],
                "value": result["value"],
                "datatype": result["datatype"],
                "station": result["station"],
                "attributes": result["attributes"],
            },
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}

# Function to save data to NetCDF
def save_to_netcdf(era5_data, output_file="era5_new_york.nc"):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    nc_file_path = os.path.join(OUTPUT_DIR, output_file)

    if os.path.exists(nc_file_path):
        nc_file = Dataset(nc_file_path, "a", format="NETCDF4")
        current_time_len = len(nc_file.dimensions["time"])
    else:
        nc_file = Dataset(nc_file_path, "w", format="NETCDF4")
        nc_file.createDimension("time", None)
        nc_file.createDimension("location", len(era5_data["features"]))
        nc_file.createDimension("latitude", len(lat_array))  # Add latitude dimension
        nc_file.createDimension("longitude", len(lon_array))  # Add longitude dimension
        
        current_time_len = 0

        # Create variables for time, datetime, latitude, longitude, value, and date
        times = nc_file.createVariable("time", "f8", ("time",))
        datetimes = nc_file.createVariable("datetime", "str", ("time",))
        latitudes = nc_file.createVariable("latitude", "f4", ("latitude",))  # Latitude array
        longitudes = nc_file.createVariable("longitude", "f4", ("longitude",))  # Longitude array
        values = nc_file.createVariable("value", "f4", ("location",))
        dates = nc_file.createVariable("date", "str", ("location",))

        # Assign latitude and longitude arrays to variables
        latitudes[:] = lat_array  # Assign the lat_array to the latitude variable
        longitudes[:] = lon_array  # Assign the lon_array to the longitude variable

    num_new_entries = len(era5_data["features"])
    times = nc_file.variables["time"]
    datetimes = nc_file.variables["datetime"]

    times[current_time_len:current_time_len + num_new_entries] = np.arange(current_time_len, current_time_len + num_new_entries)

    for i, feature in enumerate(era5_data["features"]):
        latitudes = nc_file.variables["latitude"]
        longitudes = nc_file.variables["longitude"]
        values = nc_file.variables["value"]
        dates = nc_file.variables["date"]
        
        latitudes[current_time_len + i] = feature["geometry"]["coordinates"][1] if feature["geometry"]["coordinates"][1] else np.nan
        longitudes[current_time_len + i] = feature["geometry"]["coordinates"][0] if feature["geometry"]["coordinates"][0] else np.nan
        values[current_time_len + i] = feature["properties"]["value"]
        dates[current_time_len + i] = feature["properties"]["date"]

        datetimes[current_time_len + i] = feature["properties"]["date"]

    nc_file.close()

# Flask route to post and process data
@app.route("/post_data", methods=["POST"])
def post_data():
    try:
        data = request.json
        if not data:
            raise ValueError("No data provided.")
        
        dataset_id = data.get("dataset_id")
        date = data.get("date")
        city_name = data.get("city_name")
        data_type = data.get("data_type")
        
        if not all([dataset_id, date, city_name, data_type]):
            raise ValueError("Missing required fields in payload.")
        
        location_id = "FIPS:36"  # Placeholder for city-based NOAA location
        noaa_data = fetch_noaa_data(dataset_id, date, location_id, data_type)
        if not noaa_data.get("results"):
            raise ValueError("No 'results' found in metadata response.")
        
        era5_data = transform_to_era5_format(noaa_data, city_name)

        output_filename = f"era5_{city_name.lower()}_{date}.nc"
        save_to_netcdf(era5_data, output_filename)

        nc_file_path = os.path.join(OUTPUT_DIR, output_filename)
        nc_file = Dataset(nc_file_path, "r", format="NETCDF4")

        times = nc_file.variables["time"][:]
        datetimes = nc_file.variables["datetime"][:]
        latitudes = nc_file.variables["latitude"][:]
        longitudes = nc_file.variables["longitude"][:]
        values = nc_file.variables["value"][:]

        times = times.astype(float)
        latitudes = latitudes.astype(float)
        longitudes = longitudes.astype(float)
        values = values.astype(float)

        saved_data = []
        for i in range(len(times)):
            datetime_value = datetimes[i]
            if isinstance(datetime_value, bytes):
                datetime_value = datetime_value.decode()

            saved_data.append({
                "datetime": datetime_value,
                "latitude": latitudes[i],
                "longitude": longitudes[i],
                "time": times[i],
                "value": values[i],
            })
        
        nc_file.close()

        return jsonify({
            "message": f"Data saved to {output_filename} in {OUTPUT_DIR}.",
            "saved_data": saved_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
