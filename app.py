from flask import Flask, request, jsonify
import datetime
import math
import xarray
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage
from matplotlib import animation
from io import BytesIO
from typing import Optional
import requests
import os
# import genecast_mini_demo2.py

app = Flask(__name__)

# NOAA API Configuration
NOAA_API_TOKEN = "qfgAteWLTqygunRnlVLnZaTAwbOXBrlT"
BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

# Initialize Google Cloud Storage Client (assuming it is used for files)
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"

# Function to get location ID by city name (example)
def get_location_id_by_city(city_name):
    city_location_map = {
        "New York": "FIPS:36",
        "Los Angeles": "FIPS:06",
        "Chicago": "FIPS:17",
    }
    return city_location_map.get(city_name)

    

# Fetch data from NOAA API
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
        data = response.json()
        # Debugging: log the available variables
        print("NOAA data response:", data)
        return data
    else:
        raise ValueError(f"Error fetching data: {response.status_code}, {response.text}")

# Transform NOAA data to ERA5-like JSON format
def transform_to_era5_format(noaa_data):
    features = []
    for result in noaa_data.get("results", []):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [None, None],
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

@app.route("/process_noaa_data", methods=["POST"])
def process_noaa_data():
    try:
        # Get the incoming request data (city_name, start_date, end_date)
        request_data = request.get_json()
        city_name = request_data.get("city_name")
        start_date = request_data.get("start_date")
        end_date = request_data.get("end_date")

        # Validate incoming data
        if not city_name or not start_date or not end_date:
            raise ValueError("City name, start date, and end date are required.")

        # Get location ID based on the city name
        location_id = get_location_id_by_city(city_name)
        if not location_id:
            raise ValueError("Invalid city name provided.")

        # Fetch NOAA data based on the provided parameters
        noaa_data_fetched = fetch_noaa_data(dataset_id="GHCND",  # Use an example dataset
                                             date=start_date,
                                             location_id=location_id,
                                             data_type="TMAX")  # Example data type, adjust as needed

        # Ensure that NOAA data is received
        if not noaa_data_fetched or 'results' not in noaa_data_fetched:
            raise ValueError("No data found for the specified parameters.")

        # Extract the relevant values from the NOAA data response
        noaa_results = noaa_data_fetched['results']
        if not noaa_results:
            raise ValueError("No valid results in NOAA data.")

        # Debugging: print the raw numerical data fetched from NOAA API
        for result in noaa_results:
            print(result)  # Show each result's data

        # Example: Let's just process the first result, but you could process all of them
        first_result = noaa_results[0]
        date = first_result['date']
        value = first_result['value']

        # For debugging: print the numerical data being processed
        print(f"Date: {date}, Value: {value}")

       # Remove the time portion of the date string (split by 'T' and take the first part)
        date = first_result['date'].split("T")[0]
        # Convert the date string into a datetime object
        current_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        # Prepare a list of future dates for the next 10 days
        future_dates = [(current_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 11)]


        predictions = []

        # Fetch data for the next 10 days and make predictions
        for future_date in future_dates:
            future_data_fetched = fetch_noaa_data(dataset_id="GHCND",  # Use an example dataset
                                                   date=future_date,
                                                   location_id=location_id,
                                                   data_type="TMAX")  # Example data type, adjust as needed
            if future_data_fetched and 'results' in future_data_fetched:
                future_result = future_data_fetched['results'][0]
                predictions.append({
                    "date": future_result['date'],
                    "value": future_result['value'],
                })

        # Return the 10-day predictions
        return jsonify({
            "status": "success",
            "predictions": predictions,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    
