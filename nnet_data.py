"""
format generated csv data into compressed numpy array format
"""

import numpy as np
import pandas as pd
import os


def load_data_csv(dir, row_freq):
    data = {}
    for _, _, files in os.walk(dir):
        for fname in files:
            _data = pd.read_csv(dir + fname).as_matrix()
            indices = []
            for i in range(_data.shape[0]):
                if 'V2' in _data[i]:
                    indices.append(i)
            _data = np.delete(_data, indices, axis=0)
            data[fname] = np.zeros((row_freq, _data.shape[1], _data.shape[0]//row_freq))
            for i in range(row_freq, _data.shape[0]+1, row_freq):
                data[fname][:, :, i//row_freq - 1] = _data[(i-row_freq):i, :]
    return data


def load_data_npy(dir):
    data = {}
    for _, _, files in os.walk(dir):
        for fname in files:
            data[fname[:-4]] = np.load(dir + fname)
    return data


def save_data(data, dir):
    for key in data.keys():
        np.save(dir + key[:-4], data[key])


if __name__ == '__main__':
    """
    print('...Loading and parsing data')
    data = load_data_csv(dir='./data/', row_freq=25)
    print('...Saving formatted data')
    save_data(data, dir='./data_formatted/')
    """
    data = load_data_npy('./data_formatted/')
    files = list(data.keys())
    print(data.keys())
    print(data[files[0]][:, :3, 0])