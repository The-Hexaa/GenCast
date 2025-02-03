from flask import Flask, request, jsonify
import urllib.parse
import urllib.request
import cdsapi
import json
import subprocess

app = Flask(__name__)

@app.route('/get_coordinates', methods=['GET'])
def get_coordinates():
    city = request.args.get('city', '')  # Get city from query parameter
    if not city:
        return jsonify({"error": "City parameter is required"}), 400
    
    try:
        url = 'http://nominatim.openstreetmap.org/search?'
        params = urllib.parse.urlencode(dict(q=city, format='json'))
        response = urllib.request.urlopen(url + params)
        data = json.loads(response.read().decode('utf-8'))[0]
        lat = float(data['lat'])
        lon = float(data['lon'])
        
        retrieve_Data(lat, lon) 
        return jsonify({"latitude": lat, "longitude": lon})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to calculate the bounding box (north, west, south, east) for a city
def get_bounding_box(lat, lon, offset=0.1):
    # Using an offset to create a bounding box around the city
    # You can modify the offset based on the scale of the city
    return [lat + offset, lon - offset, lat - offset, lon + offset]

def retrieve_Data(lat, lon):
    box = get_bounding_box(lat, lon)
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "land_sea_mask"
            "2m_temperature",
            "sea_surface_temperature",
            "mean_sea_level_pressure",
            "total_precipitation",
            "10m_v_component_of_neutral_wind",
            "10m_u_component_of_wind",
            "friction_velocity",
            "10m_v_component_of_wind",
            "geopotential",
        ],
        "year": ["2025"],
        "month": ["01"],
        "day": ["24"],
        "time": ["23:00"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [box[0], box[1], box[2], box[3]]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


if __name__ == '__main__':
    app.run(debug=True)
