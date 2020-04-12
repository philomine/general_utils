import os 
import numpy as np 
import pandas as pd 
import pickle 

def _get_variable_name(filename):
    variable_name = filename.split('.')[0]
    variable_name = variable_name.lower()
    while variable_name[0].isdigit() or variable_name[0] == ' ':
        variable_name = variable_name[1:]
    variable_name = '_'.join(variable_name.split(' '))
    return variable_name

def _get_extension(filename):
    return filename.split('.')[-1].lower()

def load_numpy(filepath):
    try:
        data = np.load(filepath)
    except ValueError:
        data = np.load(filepath, allow_pickle=True)
    return data

def load_pickle(filepath):
    return pickle.load(open(filepath, "rb"))

_read_functions = {
    'xls': pd.read_excel, 'xlsx': pd.read_excel, 
    'csv': pd.read_csv, 
    'feather': pd.read_feather,
    'parquet': pd.read_parquet,
    'npy': load_numpy,
    'pickle': load_pickle, 'pkl': load_pickle,
}

def _load_file(path, filename):
    extension = _get_extension(filename)
    read_function = _read_functions[extension]
    return read_function(path + filename)

def extractor(subfolder, var_name):
    path = './data/' + subfolder + '/'
    filenames = os.listdir(path)

    varnames_to_filenames = {}
    for filename in filenames:
        varnames_to_filenames[_get_variable_name(filename)] = filename 
    
    return _load_file(path=path, filename=varnames_to_filenames[var_name])
