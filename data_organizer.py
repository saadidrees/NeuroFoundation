#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:03:02 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

"""
Requires installation of allen sdk. Follow https://allensdk.readthedocs.io/en/latest/_static/examples/nb/visual_behavior_ophys_data_access.html

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
from collections import namedtuple
df_tuple = namedtuple('df_tuple', ['timestamps', 'data'])
import time
import re

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

from data_handler import dataExtractor


output_dir = "/home/saad/data/analyses/allen/raw/"
output_dir = Path(output_dir)
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=output_dir)     # download manifest files


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

This section will generate the following:
    dff_grand --> [[experiment]][time,units] --> rows of list are for different experiments. Each numpy array in the list gives calcium signal for all units in that experiment
    dff_mov_grand --> [[experiment]][y,x,time]  --> A numpy array within a list. Rows of list are for individual experiments
    dff_mov_grand_metadata --> [[experiment]] contains the stimulus that was shown at each time point
    ophys_exp_id_grand --> [[experiments]] contains the id of experiments corresponding to above variables. 
    exp_ids_grand --> dataframe containing meta information

ADD NUMBER OF SESSIONS TO EXTRACT? --> Or maybe do this in the step to organize dataset for training

"""
# ---- loading parameters
select_targetStructure = 'VISam'    # VISal', 'VISam', 'VISl', 'VISp
n_mice = 1
n_cells = -1        # -1 = all cells
initial_time = 100  # seconds
n_dur = -1 #2000        # seconds | -1 = till end


if n_dur>0:
    final_time = initial_time + n_dur
else:
    final_time = -1

# ---- Initialize empty lists for final data
dff_grand = []                  # [experiments][time,units]
dff_mov_grand = []              # [experiments][rows,cols,time] <-- movie
dff_mov_grand_metadata = []     # [experiments][timestamps,stim] <-- this contains the stimulus name that was shown at each timestamp. 0 indicates no stimulus
ts_grand = []                   # [experiments][timestamps] <--- corresponds to time axis in above variables
total_cells = 0
cell_id_grand = []              # [units]
exp_ids_grand = pd.DataFrame()  # metadata for each unit in the dataset
ophys_exp_id_grand = []         # [experiments]
mouse_id_grand = []             # all mouse ids 


# ---- 1. select target structure
targetStructures_all = np.unique(table_ophysExps.targeted_structure)         # All the target structures in ABO
exp_table_subArea = table_ophysExps.query('targeted_structure == @select_targetStructure')

# ---- 2. select mice
# containerIds_all = np.unique(exp_table_subArea.ophys_container_id)
unique_mice = np.unique(exp_table_subArea.mouse_id.values)              #  All unique mice ids for this structure
select_miceIds = unique_mice[:n_mice]                                    # Select ids for n_mice only

containerIds_nmice = np.unique(exp_table_subArea.query('mouse_id in @select_miceIds').ophys_container_id)   # All container ids corresponding to target structure and n_mice

# ---- 3 Loop over containers and then experiments
# select_container = containerIds_nmice[0]
for select_container in containerIds_nmice:
    
    # Select all experiments within a container 
    exp_table_subset = exp_table_subArea.query('ophys_container_id == @select_container')# and targeted_structure == @select_targetStructure')
    ophys_exp_ids_inContainer = exp_table_subset.index.values
    
    
    # loop over experiments in container and get list of all units in that container
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
    cell_id_grand = cell_id_grand + cell_id_unique.tolist()
    print('num of cells: %d'%len(cell_id_unique))
    total_cells = total_cells + len(cell_id_unique)
    
    # Following two loops build the exp_ids_perCell dataframe which stores info for each unit id
    # For each unit id, it stores the experiment ids the unit was recorded in, the container id and the mouse id
    loc = np.zeros((cell_id_unique.shape[0],len(ophys_exp_ids_inContainer)),dtype='bool')
    for i in range(len(cell_specimen_ids_all)):
        rgb = cell_specimen_ids_all[i]
        loc[:,i] = np.isin(cell_id_unique,rgb)      # row is for each unit, column tells which exp ids this unit belongs to
        
    exp_ids_perCell = pd.DataFrame()
    for i in range(len(cell_id_unique)):
        rgb = ophys_exp_ids_inContainer[loc[i,:]]
        mouse_id = table_ophysExps.loc[rgb].mouse_id
        df = pd.DataFrame({'cell_id':cell_id_unique[i],'exp_id':rgb,'container_id':select_container,'mouse_id':mouse_id})
        exp_ids_perCell = pd.concat((exp_ids_perCell,df),axis=0)
    
    exp_ids_grand = pd.concat((exp_ids_grand,exp_ids_perCell),axis=0)
    
    
    # Inner loop over experiments. This should append 'sessions'
    # ophys_exp_id = ophys_exp_ids_inContainer[0]
    for ophys_exp_id in ophys_exp_ids_inContainer:
        dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)
        ophys_framerate = dataset.metadata['ophys_frame_rate']
        
        cell_id_toExtract = np.array(exp_ids_perCell.query('exp_id == @ophys_exp_id').cell_id)      # cell ids in this experiment
        # cell_idxInFull = np.intersect1d(cell_id_toExtract,cell_id_unique,return_indices=True)[2]    # location of these cell ids in the master cell id list which is given by cell_id_unique
        
        if final_time == -1:
            timestamps = dataset.ophys_timestamps
            final_time = np.floor(timestamps[-1])
        dataDict = dataExtractor.get_dataDict(dataset,cell_id_toExtract,initial_time,final_time)     # makes a dictionary containing dff, licks, pupil etc etc for this experiment
        
        dff = dataDict['dff'].data                  # calcium signal [time,units]
        ts = dataDict['dff'].timestamps.values      # timestamps corresponding to dff
        rois = dataDict['rois']                     # rois for each unit [y,x,units]
        
        stimLabel_dff = np.zeros((dff.shape[0]),dtype='object')     # this will contain the stimilus name for each time step
        unique_stimuli = [stimulus for stimulus in dataDict['stim_pres'].image_name.unique()]
        i=0
        for i in range(len(unique_stimuli)):
            selectedStim = unique_stimuli[i]
            idx_selectedStim = dataDict['stim_pres'].image_name==selectedStim
            ts_selectedStim = np.array(dataDict['stim_pres'].loc[idx_selectedStim,['start_time','end_time']])
            
            a = np.diff(ts_selectedStim,axis=0)
            thresh_gaps = 20    # seconds
            b = np.where(a[:,0]>thresh_gaps)[0]
            c_start = np.concatenate((np.array([0]),b+1))
            c_end = np.concatenate((np.array([0]),b))
                
            selectedStim_startEnd = np.array([ts_selectedStim[c_start[:-1],0],ts_selectedStim[c_end[1:],1]]).T
            j=0
            for j in range(selectedStim_startEnd.shape[0]):
                idx_ts = np.logical_and(ts>selectedStim_startEnd[j,0],ts<selectedStim_startEnd[j,1])
                stimLabel_dff[idx_ts] = selectedStim
    
                
        roi_movie = np.zeros((rois.shape[0],rois.shape[1],dff.shape[0]))
        
        i=0
        for i in range(dff.shape[-1]):  # loop over num of cells
            idx = np.where(rois[:,:,i])
            roi_movie[idx[0],idx[1],:] = dff[:,i]
        
        df_movie_meta = pd.DataFrame(ts,columns=['timestamps'])
        df_movie_meta['stim'] = stimLabel_dff
        
        dff_grand.append(dff)                     # [[experiments]][time,units]
        dff_mov_grand.append(roi_movie)           # [[experiments]][y,x,time]
        dff_mov_grand_metadata.append(df_movie_meta) # [[experiments]]
        ts_grand.append([ts])                       # [[experiments]][time]
        ophys_exp_id_grand.append(ophys_exp_id)
        mouse_id = np.unique(exp_ids_grand.query('exp_id==@ophys_exp_id').mouse_id.astype('int'))[0]
        mouse_id_grand.append(mouse_id)
            

# %% Train Val dsets for wav2vec
"""
The sampling rate is 11 Hz. So interval time of ~91 ms. If i take 1000 samples in a sequence
then it will be 91 s per sequence. Which I think is fine. 
"""
from data_handler import utils


unique_stimuli = np.zeros((0),dtype='object')
i=0
for i in range(len(dff_mov_grand_metadata)):
    rgb = np.array(dff_mov_grand_metadata[i].stim.unique())
    unique_stimuli = np.concatenate((unique_stimuli,rgb),axis=0)
unique_stimuli[unique_stimuli==0]='0'
unique_stimuli = np.unique(unique_stimuli)
print(unique_stimuli)


stim_val = 'im115_r'
downscale_fac = 4
context_len = 80

dset_train = []
dset_val = []

i=0
for i in range(len(dff_mov_grand)):
    mov = dff_mov_grand[i]
    mov = mov[::downscale_fac,::downscale_fac,:]
    mov = np.moveaxis(mov,-1,0)
    
    idx_train = np.array(dff_mov_grand_metadata[i].stim != stim_val)
    idx_val = ~idx_train
    
    mov_train = mov[idx_train]
    mov_val = mov[idx_val]
    
    dict_metadata = dict(mouse_id=mouse_id_grand[i],
                         ophys_exp_id=ophys_exp_id_grand[i])
    temp = utils.chunker(mov_train,context_len,dict_metadata=dict_metadata)
    dset_train = dset_train+temp
    
    temp = utils.chunker(mov_val,context_len,dict_metadata=dict_metadata)
    dset_val = dset_val+temp


# dset_train = dset[:-5]
# # dset_train = dset[:-10]
dset_val = dset_val[-5:]


