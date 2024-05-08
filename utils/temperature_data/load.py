import numpy as np
import glob
import netCDF4

def load_nc_files(datapath):
    # Create a pattern to match all .nc files in the directory
    file_pattern = f"{datapath}/*.nc"
    # Use glob to find all files matching the pattern
    files = glob.glob(file_pattern)
    datasets = []
    
    # Loop through all files and load them using netCDF4
    for file in files:
        dataset = netCDF4.Dataset(file)
        datasets.append(dataset)
        # You can perform operations on 'dataset' here if needed
    
    return datasets

def get_time_data(dataset):
    # Access the time variable
    time_var = dataset.variables["time"]
    # Extract the data as a numpy array
    time_data = time_var[:]
    return time_data

def get_lon_data(dataset):
    # Access the longitude variable, degrees east
    lon_var = dataset.variables["lon"]
    # Extract the data as a numpy array
    lon_data = lon_var[:]
    return lon_data

def get_lat_data(dataset):
    # Access the latitude variable, degrees north
    lat_var = dataset.variables["lat"]
    # Extract the data as a numpy array
    lat_data = lat_var[:]
    return lat_data

def get_temperature_data(dataset):
    # Access the temperature variable
    temp_var = dataset.variables["Temperature_Air_2m_Mean_24h"]
    
    # Extract the data as a numpy array
    temp_data = temp_var[:]

    # Mask the fill values and convert them to NaN
    temp_data_masked = np.ma.masked_equal(temp_data, temp_var._FillValue)
    temp_data_filled = temp_data_masked.filled(np.nan)  # Replace masked values with NaN
    
    return temp_data_filled