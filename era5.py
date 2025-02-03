import cdsapi
from geopy.geocoders import Nominatim
import os
import netCDF4

# Function to get latitude and longitude from city name
def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="city_locator")
    location = geolocator.geocode(city_name)
    
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError("City not found")

# Function to calculate the bounding box (north, west, south, east) for a city
def get_bounding_box(lat, lon, offset=0.1):
    # Using an offset to create a bounding box around the city
    # You can modify the offset based on the scale of the city
    return [lat + offset, lon - offset, lat - offset, lon + offset]

# Function to download ERA5 data for a given city
def download_era5_data_for_city(city_name):
    try:
        # Get city coordinates
        lat, lon = get_city_coordinates(city_name)
        
        # Calculate the bounding box
        area = get_bounding_box(lat, lon)
        
        # Define the dataset and request parameters with more variables
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "sea_surface_temperature",
                "surface_pressure",
                "total_precipitation",
                "geopotential_at_surface",  # Adding geopotential variable
                "temperature",               # Adding temperature at different levels
                "vertical_velocity",         # Adding vertical velocity
                "v_component_of_wind",       # Adding wind component
                "geopotential",              # Adding geopotential at different levels
                "sea_surface_height",        # Adding sea surface height
                "mean_sea_level_pressure",   # Adding mean sea level pressure
                "low_cloud_cover"            # Adding low cloud cover
            ],
            "year": ["2025"],
            "month": [
                "01"
            ],
            "day": [
                "01"
            ],
            "time": [
                "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", 
                "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", 
                "22:00", "23:00"
            ],
            "format": "netcdf",
            "area": area  # Set area dynamically based on city
        }

        # Initialize the CDS client
        client = cdsapi.Client()

        # Set the output directory for download
        output_directory = os.path.join(os.getcwd(), 'data')
        os.makedirs(output_directory, exist_ok=True)  # Create .data directory if it doesn't exist
        
        output_file_path = os.path.join(output_directory, "output_data.nc")

        # Make the request and initiate the download
        client.retrieve(dataset, request, output_file_path)
        print(f"Download initiated for {city_name}. Data will be saved to '{output_file_path}'.")

    except Exception as e:
        print(f"Error downloading data: {str(e)}")


# Example usage
download_era5_data_for_city("New York")
