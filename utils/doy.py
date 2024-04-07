import numpy as np 
import requests
import os

def get_doys_dict_test(dataroot:str, npydoy: str='breizhcrops_frh04_2017_doys.npy'):
    """ loads the dictionary of doys from the npy file, if it does not exist, it downloads it from the internet"""
    filepath = os.path.join(dataroot, npydoy)
    if not os.path.exists(filepath):
        url = "https://elects.s3.eu-central-1.amazonaws.com/"+npydoy
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
    doys_dict = np.load(filepath, allow_pickle=True).flat[0]
    return doys_dict

def get_doy_stop(stats, doys_dict, t_stop_key="t_stop", approximated=True):
    """ returns the day of year at which the model stops for each sample in the stats dictionary
    if approximated is True: 
        - the doys_dict is a dictionary of approximated doys, 
        - the keys are the length of the corresponding doy,
        - the values are the approximated doys (from get_approximated_doy_function)
        - the ids are the lengths of the sequences
    if approximated is False: 
        - the doys_dict is a dictionary of true doys (from the test set)
        - the keys are the ids of the samples
        - the values are the true doys of the samples
        - the ids are the ids of the samples
    """
    doy_stop = []
    if approximated:
        ids_list = stats["seqlengths"]
    else:
        ids_list = stats["ids"][:,0]

    if t_stop_key=="t_stop":
        t_stop_list = stats["t_stop"][:,0]
    elif t_stop_key=="t_stop_max_prob": 
        t_stop_list = stats["t_stop_max_prob"]
    else:
        raise ValueError(f"t_stop_key {t_stop_key} not recognized")
    
    for id, t_stop in zip(ids_list, t_stop_list):
        doys = doys_dict[id]
        if t_stop>=len(doys):
            t_stop = len(doys)-1
        doy_stop.append(doys[t_stop-1])
    doy_stop = np.array(doy_stop)
    return doy_stop

def get_closest_length_key(length, doys_dict):
    """ Given a length, get the sequence index in doys_dict that has the closest length"""
    keys = list(doys_dict.keys())
    closest_key = keys[0]
    closest_length = len(doys_dict[keys[0]])
    for key in keys:
        if abs(len(doys_dict[key])-length) < abs(closest_length-length):
            closest_key = key
            closest_length = len(doys_dict[key])
            if closest_length == length:
                return closest_key
    return closest_key

def get_approximated_doy(length, doys_dict):
    """
    Return the appoximated doy, for a given length. The approximated doy is the one that has the closest length to the given length in doys_dict. 
    If the length is greater than the length of the longest sequence (104), return the longest sequence.
    """
    closest_key = get_closest_length_key(length, doys_dict)
    approximated_doy = doys_dict[closest_key]
    if len(approximated_doy)<length:
        pad = np.arange(approximated_doy[-1], 365, (365-approximated_doy[-1])/(length-len(approximated_doy))).astype(int)
        approximated_doy = np.append(approximated_doy, pad)
    elif len(approximated_doy)>length: 
        approximated_doy = approximated_doy[:length+1]
    return approximated_doy

def create_sorted_doys_dict_test(doys_dict):
    """ created a dictionnary where the key is the length of the doy and the value is the approximated doy."""
    sorted_doys_dict = {}
    keys = np.arange(49, 105, 1)
    for key in keys: 
        approximated_doy = get_approximated_doy(key, doys_dict)
        sorted_doys_dict[key] = approximated_doy
    return sorted_doys_dict

def get_approximated_doys_dict(lengths: list, length_sorted_doys_dict: dict):
    """ 
    Return the approximated doys dict, from the lengths of the validation set.
    The length_sorted_doys_dict is a dictionary where the key is the length of the doy and the value is the approximated doy.
    """
    doys_dict = {}
    for idx, length in enumerate(lengths):
        approximated_doy = length_sorted_doys_dict[length]
        doys_dict.update({idx: approximated_doy})
    return doys_dict