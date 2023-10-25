#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:33:02 2023

@author: saad
"""

"""
Notes:
    1. Add option for zero hidden layer, i.e. directly a FC layer for classificiation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

class probe_hidden (nn.Module):
    def __init__(self,config_probe,layer_id):
        super().__init__()
        
        if layer_id ==0 and len(config_probe.nunits_hidden)>0:
            self.hidden = nn.Linear(config_probe.dim_inpFeatures,config_probe.nunits_hidden[layer_id]).double()
            self.batchnorm = nn.BatchNorm1d(config_probe.nunits_hidden[layer_id],affine=True).double()
            self.activation = nn.GELU()      

        elif layer_id < len(config_probe.nunits_hidden):
            self.hidden = nn.Linear(config_probe.nunits_hidden[layer_id-1],config_probe.nunits_hidden[layer_id]).double()
            self.batchnorm = nn.BatchNorm1d(config_probe.nunits_hidden[layer_id],affine=True).double()
            self.activation = nn.GELU()      

        elif layer_id >= len(config_probe.nunits_hidden):
            if len(config_probe.nunits_hidden)==0:      # No hidden layers. Only output layer
                self.hidden = nn.Linear(config_probe.dim_inpFeatures,config_probe.nunits_out).double()
            else:
                self.hidden = nn.Linear(config_probe.nunits_hidden[-1],config_probe.nunits_out).double()
            self.batchnorm = nn.BatchNorm1d(config_probe.nunits_out,affine=True).double()
            self.activation = nn.Softmax(dim=1)
            

        
    def forward(self,hidden_states):
        hidden_states = self.hidden(hidden_states).double()
        hidden_states = self.batchnorm(hidden_states).double()    
        hidden_states = self.activation(hidden_states).double()           
        return hidden_states
       
    
class ProbeModel (nn.Module):
    def __init__(self,probe_layers):
        super().__init__()
        self.probe_layers = nn.ModuleList(probe_layers)
        
    def forward(self,hidden_states):
        for probe_layers in self.probe_layers:
            hidden_states = probe_layers(hidden_states)      
        return hidden_states


def l2_reg(kernel_reg,mdl):
    l2_pen = 0
    for name, parameter in mdl.named_parameters():
        p_regex = re.compile(r'hidden.weight')
        rgb = p_regex.search(name)
        if rgb != None:
            l2_pen = l2_pen + (kernel_reg*(parameter**2).sum())
            
    return l2_pen


def get_layerTempOutputDim(context_len,conv_kernel,conv_stride):
    input_length = context_len
    for i in range(len(conv_kernel)):
        input_length = np.floor((input_length - conv_kernel[i])/conv_stride[i])+1        

    
    return input_length