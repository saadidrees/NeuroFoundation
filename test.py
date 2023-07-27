#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:03:02 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

"""
- Experiment table contains ophys experiment ids as well as associated metadata with them (cre_line, session_type, project_code, etc). 
    This table gives you an overview of what data at the level of each experiment is available. The term experiment is used to describe one 
    imaging plane during one session. For sessions that are imaged using mesoscope (equipment_name = MESO.1), there will be up to 4 experiments 
    associated with that sessions (2 imaging depths by 2 visual areas). Across sessions, the same imaging planes or experiments are linked using 
    ophys_container_id. For sessions that are imaged using scientifica (equipment_name = CAMP#.#), there will be only 1 experiment which are 
    similarly linked across different session types using ophys_container_id.
- Ophys session table is similar to experiment table but it is a higher level overview of the data details. 
    It groups imaging sessions using ophys_session_id and provides metadata associated with those sessions.
- Behavior session table contains metadata related to animals' training history as well as their behavior during ophys imaging sessions. 
    The table is organized using behavior_session_id. Behavior sessions that were also imaging sessions have ophys ids assosiated with them.

- Session: Data collected in a single continous recording
- Experiment: imaging of a single plane per session. I think this is the lowest level of recording. 
              For multi-plane imaging experiments, there can be up to 8 imaging planes (8 experiments) per session.
- Sesstion types: Activity of single neurons across multiple stimuli set (that would be across multiple sessions then).
- Container: Collection of imaging sessions for a given population of cells, belonging to a single imaging plane,
             measured across days. --> I THINK THIS IS WHAT WE NEED FOR MODELING
             
Each mouse can have one or more containers, each representing a unique imaging plane 'experiment' that has been targeted across multiple
recording 'sessions', under differetn behavioral and sensory conditions 'session types'.
"""


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import namedtuple
df_tuple = namedtuple('df_tuple', ['timestamps', 'data'])


import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc

from data_handler import dataExtractor


DOWNLOAD_COMPLETE_DATASET = False 

output_dir = "/home/saad/data/analyses/allen/raw/"
output_dir = Path(output_dir)
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=output_dir)     # download manifest files

"""
dset_generator
The ideal dset_generator should be able to pull out:
    1. Data for all cells for N mice
    2. N cells
    3. Time window
    4. Cells within a container exceeding threshold for number of experiments
    
The final dset should look like:
    - dff: [time,experiments,cells]
    - stimulus: [time,experiments]
    - speed: [time,experiments]
    - pupil: [time,experiments]
"""

table_behavior = cache.get_behavior_session_table()
table_ophysSession = cache.get_ophys_session_table()
table_ophysExps = cache.get_ophys_experiment_table()


# %% Load, extract and organize data 
"""
1. Select cell/area type
2. Get container ids for that cell/area type
3. for N mice
3. Pick a container
4. Get exp ids within that container
5. Loop over experiments within the container and get the ids of cells that are
   common across sessions (or not if you set threshold to 0)

This section will create a set of arrays where each row is for a difference cell.
Each cell will contain a list of arrays, representing sessions. So if a cell was recorded
over multiple experiments, it will have multiple arrays. Within each array, data is represented
with the dimension 0 being time. 
[[[cells]]][[sessions]][data]


ADD NUMBER OF SESSIONS TO EXTRACT? --> Or maybe do this in the step to organize dataset for training

"""
# Define selecting params
select_targetStructure = 'VISam'    # VISal', 'VISam', 'VISl', 'VISp
n_mice = 1
n_cells = -1
initial_time = 100  # seconds
n_dur = -1 #2000        # seconds
thresh_numSessions = 3
bin_ms = -1#16 


if n_dur>0:
    final_time = initial_time + n_dur
else:
    final_time = -1

# Initialize empty variables where the number of rows will eventually correspond to number of cells
dff_grand = []
dff_mov_grand = []
ts_grand = []
total_cells = 0
cell_id_grand = []
exp_ids_grand = pd.DataFrame()


#1
targetStructures_all = np.unique(table_ophysExps.targeted_structure)
exp_table_subArea = table_ophysExps.query('targeted_structure == @select_targetStructure')

#2
containerIds_all = np.unique(exp_table_subArea.ophys_container_id)
unique_mice = np.unique(exp_table_subArea.mouse_id.values)
select_miceIds = unique_mice[:n_mice]

containerIds_nmice = np.unique(exp_table_subArea.query('mouse_id in @select_miceIds').ophys_container_id)

# Will have to loop over containers and then experiments
#3
select_container = containerIds_nmice[0]
for select_container in containerIds_nmice:
    
    #4
    exp_table_subset = exp_table_subArea.query('ophys_container_id == @select_container')# and targeted_structure == @select_targetStructure')
    ophys_exp_ids_inContainer = exp_table_subset.index.values
    
    
    #5 loop over experiments in container and get list of intersection cells
    cell_id_list = np.array([],dtype='int64')
    cell_specimen_ids_all = []
    ophys_exp_id = ophys_exp_ids_inContainer[0]
    for i in range(ophys_exp_ids_inContainer.shape[0]):
        ophys_exp_id = ophys_exp_ids_inContainer[i]
        dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)
        cell_specimen_table = dataset.cell_specimen_table
        cell_specimen_ids = cell_specimen_table.index.values
        cell_id_list = np.append(cell_id_list,cell_specimen_ids.flatten())
        cell_specimen_ids_all.append(cell_specimen_ids)
    
    cell_id_unique,cell_id_counts = np.unique(cell_id_list,return_counts=True)
    idx_cells = cell_id_counts>thresh_numSessions
    cell_ids_full = cell_id_unique[idx_cells]
    cell_id_grand = cell_id_grand + cell_id_unique.tolist()
    print('num of cells: %d'%idx_cells.sum())
    total_cells = total_cells + idx_cells.sum()
    
    # Get experiment ids / sessions where all these cells are present
    i=0
    loc = np.zeros((cell_ids_full.shape[0],len(ophys_exp_ids_inContainer)),dtype='bool')
    for i in range(len(cell_specimen_ids_all)):
        rgb = cell_specimen_ids_all[i]
        loc[:,i] = np.isin(cell_ids_full,rgb)
        countPerExp = np.sum(np.isin(cell_ids_full,rgb))
        print(countPerExp)
        
    i=1
    exp_ids_perCell = pd.DataFrame()
    for i in range(len(cell_ids_full)):
        rgb = ophys_exp_ids_inContainer[loc[i,:]]
        mouse_id = table_ophysExps.loc[rgb].mouse_id
        df = pd.DataFrame({'cell_id':cell_ids_full[i],'exp_id':rgb,'container_id':select_container,'mouse_id':mouse_id})
        exp_ids_perCell = pd.concat((exp_ids_perCell,df),axis=0)
    
    exp_ids_grand = pd.concat((exp_ids_grand,exp_ids_perCell),axis=0)
    
    
    
    # Inner loop over experiments. This should append 'sessions'
    ophys_exp_id = ophys_exp_ids_inContainer[0]
    for ophys_exp_id in ophys_exp_ids_inContainer:
        dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)
        ophys_framerate = dataset.metadata['ophys_frame_rate']
        
        cell_id_toExtract = np.array(exp_ids_perCell.query('exp_id == @ophys_exp_id').cell_id)
        cell_idxInFull = np.intersect1d(cell_id_toExtract,cell_ids_full,return_indices=True)[2]
        
        if final_time == -1:
            timestamps = dataset.ophys_timestamps
            final_time = np.floor(timestamps[-1])
        dataDict = dataExtractor.get_dataDict(dataset,cell_id_toExtract,initial_time,final_time)
        
        dff = dataDict['dff'].data
        ts = dataDict['dff'].timestamps.values
        rois = dataDict['rois']
        
        roi_movie = np.zeros((rois.shape[0],rois.shape[1],dff.shape[0]))
        i=0
        for i in range(dff.shape[-1]):  # loop over num of cells
            idx = np.where(rois[:,:,i])
            roi_movie[idx[0],idx[1],:] = dff[:,i]

        
        dff_grand.append([dff])                     # [[[cells]]][[sessions]][data]
        dff_mov_grand.append([roi_movie])           # [[[cells]]][[sessions]][data]
        ts_grand.append([ts])                       # [[[cells]]][[sessions]][data]
            

# %% Batch stuff for wav2vec
"""
The sampling rate is 11 Hz. So interval time of ~91 ms. If i take 1000 samples in a sequence
then it will be 91 s per sequence. Which I think is fine. 
"""
from data_handler import utils
# from skimage.transform import rescale, resize, downscale_local_mean

downscale_fac = 4
chunk_size = 500

dset = []

i=0
for i in range(len(dff_mov_grand)):
    rgb = dff_mov_grand[i][0]
    rgb = rgb[::downscale_fac,::downscale_fac,:]
    rgb = np.moveaxis(rgb,-1,0)
    
    temp = utils.chunker(rgb,chunk_size)
    dset = dset+temp

dset_train = dset[:25]
# dset_train = dset[:-10]
dset_val = dset[-5:]



# %% for wav2vec2 old
from skimage.transform import rescale, resize, downscale_local_mean
from datasets import DatasetDict

mov_train = dff_mov_grand[0][0][:,:,:100]
mov_train = np.moveaxis(mov_train,-1,0)
mov_train = downscale_local_mean(mov_train,factors=(1,2,2))
mov_train = dict(input_values=mov_train)

mov_val = dff_mov_grand[5][0][:,:,:50]
mov_val = np.moveaxis(mov_val,-1,0)
mov_val = downscale_local_mean(mov_val,factors=(1,2,2))
mov_val = dict(input_values=mov_val)


mov_train = [DatasetDict(mov_train)]
mov_val = [DatasetDict(mov_val)]

# %%
batch_start = np.arange(0,len(dset),4)
i = 0
for i in range(len(dset)):
    rgb = dset[batch_start[i]:batch_start[i+1]]
    a1 = rgb[0]['input_values'].shape[0]
    a2 = rgb[1]['input_values'].shape[0]
    a3 = rgb[2]['input_values'].shape[0]
    a4 = rgb[3]['input_values'].shape[0]
    
    a = np.array([a1,a2,a3,a4])
    assert np.all(a == 500)

# %% Prepare train/test dataset
from data_handler import utils
from skimage.transform import rescale, resize, downscale_local_mean

temporal_window = 11
train_X = dff_mov_grand[0][0][:,:,:1024]
train_X = np.moveaxis(train_X,-1,0)
train_X = downscale_local_mean(train_X,factors=(1,2,2))

train_X = utils.rolling_window(train_X,temporal_window)
train_y = np.squeeze(train_X[:,0,100,:])
train_X = train_X[:,:10]

train_X = torch.tensor(train_X,dtype=torch.float32)
train_y = torch.tensor(train_y,dtype=torch.float32)


# %% Feat Encoder

"""
CLASStorch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary




class fe_cnn2d(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.c1=32;  self.k1=8; self.s1=4        
        self.c2=64;  self.k2=7; self.s2=2
        self.c3=128; self.k3=5; self.s3=1;
        self.c4=256; self.k4=3; self.s4=1;
        self.c5=512; self.k5=3; self.s5=1
        self.nout=256;

        self.conv1 = nn.Conv2d(10,self.c1,kernel_size=(self.k1,self.k1),stride=self.s1)
        self.ln1 = nn.LayerNorm([self.c1,63,63])
        self.act1 = nn.GELU()
            
        self.mp1 = nn.MaxPool2d(kernel_size=(3,3),stride=2)
            
        self.conv2 = nn.Conv2d(self.c1,self.c2,kernel_size=(self.k2,self.k2),stride=self.s2)
        self.ln2 = nn.LayerNorm([self.c2,13,13])
        self.act2 = nn.GELU()
            
        self.conv3 = nn.Conv2d(self.c2,self.c3,kernel_size=(self.k3,self.k3),stride=self.s3)
        self.ln3 = nn.LayerNorm([self.c3,9,9])
        self.act3 = nn.GELU()
    
        self.conv4 = nn.Conv2d(self.c3,self.c4,kernel_size=(self.k4,self.k4),stride=self.s4)
        self.ln4 = nn.LayerNorm([self.c4,7,7])
        self.act4 = nn.GELU()
    
        self.conv5 = nn.Conv2d(self.c4,self.c5,kernel_size=(self.k5,self.k5),stride=self.s5)
        self.ln5 = nn.LayerNorm([self.c5,5,5])
        self.act5 = nn.GELU()
        
        self.fl1 = nn.Flatten()
        
        self.fc1 = nn.Linear(self.c5*5*5,self.nout)
        
        
    def forward(self,x):
        # print(x.size())
        y = x
       
        y = self.conv1(y)
        y = self.ln1(y)
        y = self.act1(y)
        
        y = self.mp1(y)

        
        y = self.conv2(y)
        y = self.ln2(y)
        y = self.act2(y)

        
        y = self.conv3(y)
        y = self.ln3(y)
        y = self.act3(y)

        y = self.conv4(y)
        y = self.ln4(y)
        y = self.act4(y)


        y = self.conv5(y)
        y = self.ln5(y)
        y = self.act5(y)
        
        
        y = self.fl1(y)

        y = self.fc1(y)

        out = y
        # print(out.size())

        
        return out
    

    
mdl = fe_cnn2d()
summary(mdl,(1013,10,256,256))

loss_fn = nn.MSELoss()
optimizer = optim.Adam(mdl.parameters(), lr=0.001)
n_epochs=2
batch_size = 32
batches_per_epoch = train_X.shape[0]// batch_size
epoch=0
i=0
train_loss = []
train_acc = []
test_acc = []

for epoch in range(n_epochs):
    for i in range(batches_per_epoch):
        start = i * batch_size
        # take a batch
        Xbatch = train_X[start:start+batch_size]
        ybatch = train_y[start:start+batch_size]
        # forward pass
        y_pred = mdl(Xbatch)
        loss = loss_fn(y_pred, ybatch)
        acc = (y_pred.round() == ybatch).float().mean()
        # store metrics
        train_loss.append(float(loss))
        train_acc.append(float(acc))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        
        
# %% wav2vec2

from models.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model

configuration = Wav2Vec2Config()
mdl = Wav2Vec2Model(configuration)


                
        
