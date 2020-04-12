import os

import numpy as np
import pandas as pd
import time

import datetime



def clear_terminal():
    os.system('cls' if os.name=='nt' else 'clear')

def parent_dir(path):
    if path[-1] == '\\':
        path = path[:-1]
    path = path.split('\\')
    path = path[:-1]
    path = '\\'.join(path) + '\\'
    return path

# Medium article "Dynamically add a method to a class" by Michael Garod
# https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
def add_method(cls):
    '''
    Add a method to a class.

    :Example: 
    # Adding method foo() to class 
    @add_method(A)
    def foo():
        print('hello world!')
    # Method foo() can still be used on its own (doesn't need self attribute).
    '''
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

def elapsed_time(t0):
    return str(datetime.timedelta(seconds=time.time() - t0))[:-7]