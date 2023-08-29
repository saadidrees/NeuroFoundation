#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:22:07 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import numpy as np


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
 
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if window > 0:
        if time_axis == 0:
            array = array.T
    
        elif time_axis == -1:
            pass
    
        else:
            raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
        assert window < array.shape[-1], "`window` is too long."
    
        # with strides
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    
        if time_axis == 0:
            return np.rollaxis(arr.T, 1, 0)
        else:
            return arr
    else:
        # arr = arr[:,np.newaxis,:,:]
        return array
                 
def unroll_data(data,time_axis=0,rolled_axis=1):
    rgb = data[0]
    rgb = np.concatenate((rgb,data[1:,data.shape[1]-1,:]),axis=0)
    # rgb = np.concatenate((rgb,data[-1:,0,:]),axis=0)
    return rgb


def chunker(X,chunk_size,truncate=True,dict_metadata=None):
    data_list = []
    counter = 0
    batch_startIdx = np.arange(0, X.shape[0], chunk_size)
    for cbatch in batch_startIdx:
        temp = []
        counter+=1
        # print('chunk %d of %d'%(counter,len(batch_startIdx)))
        if truncate==True:
            if cbatch + chunk_size < X.shape[0]:
                temp = X[cbatch:(cbatch + chunk_size)]
        else:
            temp = X[cbatch:(cbatch + chunk_size)]
        
        if len(temp)>0:
            temp = dict(input_values=temp)
            if dict_metadata is not None:
                temp = temp | dict_metadata
                if 'stiminfo' in temp:
                    temp['stiminfo'] = temp['stiminfo'].iloc[cbatch:(cbatch + chunk_size)]
            data_list.append(temp)
        
    return data_list
