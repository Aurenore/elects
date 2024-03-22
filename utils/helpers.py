import numpy as np 
import requests
import os
# TO DO 

def get_doys_dict(npydoy: str='breizhcrops_frh04_2017_doys.npy'):

    npydoy = 'breizhcrops_frh04_2017_doys.npy'

    # Check if the file does not exist and download it
    if not os.path.exists(npydoy):
        url = "https://elects.s3.eu-central-1.amazonaws.com/breizhcrops_frh04_2017_doys.npy"
        response = requests.get(url)
        with open(npydoy, 'wb') as f:
            f.write(response.content)

    # Load the numpy file
    doys_dict = np.load(npydoy, allow_pickle=True)
    return doys_dict

def get_doy_stop(stats, npy_doy: str='breizhcrops_frh04_2017_doys.npy'):
    doy_stop = []
    doys_dict = get_doys_dict(npy_doy)
    for id, t_stop in zip(stats["ids"][:,0], stats["t_stop"][:,0]):
        doys = doys_dict.flat[0][id]
        doy_stop.append(doys[t_stop-1])
    doy_stop = np.array(doy_stop)
    return doy_stop