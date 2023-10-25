#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:26:07 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


def get_modelParamsN(model):
    state_dict = model.state_dict()
    n_params = np.array([0],dtype='int64')
    for key in state_dict.keys():
        n_params = n_params+np.array(state_dict[key].nelement())


"""
14 million params
"""