#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:42:28 2023

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

# %% Get experimental lists
behavior_session_table = cache.get_behavior_session_table()
ophys_session_table = cache.get_ophys_session_table()
experiment_table = cache.get_ophys_experiment_table()


# %% 
"""
1. Select cell/area type
2. Get container ids for that cell/area type
3. Pick a container
4. Loop over experiments within the container and get the ids of cells that are
   common across sessions

"""

targetStructures_all = np.unique(experiment_table.targeted_structure)
select_targetStructure = 'VISam'

exp_table_subArea = experiment_table.query('targeted_structure == @select_targetStructure')

container_ids_all = np.unique(exp_table_subArea.ophys_container_id)
select_container = container_ids_all[3]

exp_table_subset = exp_table_subArea.query('ophys_container_id == @select_container and targeted_structure == @select_targetStructure')
ophys_exp_ids_inContainer = exp_table_subset.index.values


# loop over experiments in container and get list of intersection cells
cell_id_list = np.array([],dtype='int64')
cell_specimen_ids_all = []
ophys_exp_id = ophys_exp_ids_inContainer[4]
for i in range(ophys_exp_ids_inContainer.shape[0]):
    ophys_exp_id = ophys_exp_ids_inContainer[i]
    dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)
    cell_specimen_table = dataset.cell_specimen_table
    cell_specimen_ids = cell_specimen_table.index.values
    cell_id_list = np.append(cell_id_list,cell_specimen_ids.flatten())
    cell_specimen_ids_all.append(cell_specimen_ids)

cell_id_unique,cell_id_counts = np.unique(cell_id_list,return_counts=True)
thresh_numSessions = 3
idx_cells = cell_id_counts>thresh_numSessions
cell_ids_full = cell_id_unique[idx_cells]
print('num of cells: %d'%idx_cells.sum())

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
    df = pd.DataFrame({'cell_id':cell_ids_full[i],'exp_id':rgb})
    exp_ids_perCell = pd.concat((exp_ids_perCell,df),axis=0)
# exp_ids_perCell = exp_ids_perCell.set_index('cell_id')
        
# %% Select Experiment to load

dff_grand = np.zeros((0,cell_ids_full.shape[0]));dff_grand[:] = np.nan
dff_ts_grand = np.zeros((0));dff_ts_grand[:] = np.nan
speed_grand = np.zeros((0));speed_grand[:] = np.nan
speed_ts_grand = np.zeros((0));speed_ts_grand[:] = np.nan
pupil_grand = np.zeros((0));pupil_grand[:] = np.nan
pupil_ts_grand = np.zeros((0));pupil_ts_grand[:] = np.nan
licks_grand = np.zeros((0));licks_grand[:] = np.nan
licks_ts_grand = np.zeros((0));licks_ts_grand[:] = np.nan
rewards_grand = np.zeros((0));rewards_grand[:] = np.nan
rewards_ts_grand = np.zeros((0));rewards_ts_grand[:] = np.nan

initial_time = 400
final_time = 500
bin_ms=10

# ophys_exp_id = 957759570
ophys_exp_id = ophys_exp_ids_inContainer[1]
for ophys_exp_id in ophys_exp_ids_inContainer:
    dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)
    
    
    cell_id_toExtract = np.array(exp_ids_perCell.query('exp_id == @ophys_exp_id').cell_id)
    cell_idxInFull = np.intersect1d(cell_id_toExtract,cell_ids_full,return_indices=True)[2]
    
    dataDict = dataExtractor.get_dataDict(dataset,cell_id_toExtract,initial_time,final_time)

    dff_orig = dataDict['dff']
    events_orig = dataDict['events']
    stim_pres_orig = dataDict['stim_pres']
    stimulus_presentations_orig = dataDict['stimulus_presentations']
    speed_orig = dataDict['speed']
    pupil_orig = dataDict['pupil']
    licks_orig = dataDict['licks']
    rewards_orig = dataDict['rewards']
    
    dff = dataExtractor.upsample(dff_orig,bin_ms,initial_time,final_time)
    events = dataExtractor.upsample(events_orig,bin_ms,initial_time,final_time)
    speed = dataExtractor.upsample(speed_orig,bin_ms,initial_time,final_time)
    pupil = dataExtractor.upsample(pupil_orig,bin_ms,initial_time,final_time)
    licks = dataExtractor.upsample(licks_orig,bin_ms,initial_time,final_time)
    rewards = dataExtractor.upsample(rewards_orig,bin_ms,initial_time,final_time)
    
    rgb = dff.data
    temp = np.zeros((rgb.shape[0],cell_ids_full.shape[0]));temp[:]=np.nan
    temp[:,cell_idxInFull] = rgb
    
    dff_grand = np.concatenate((dff_grand,temp),axis=0)
    dff_ts_grand = np.concatenate((dff_ts_grand,np.array(dataDict['dff'].timestamps)),axis=0)
    
    speed_grand = np.concatenate((speed_grand,np.array(dataDict['speed'].data)),axis=0)
    speed_ts_grand = np.concatenate((speed_grand,np.array(dataDict['speed'].timestamps)),axis=0)
    
    pupil_grand = np.concatenate((pupil_grand,np.array(dataDict['pupil'].data)),axis=0)
    pupil_ts_grand = np.concatenate((pupil_grand,np.array(dataDict['pupil'].timestamps)),axis=0)
    
    licks_grand = np.concatenate((licks_grand,np.array(dataDict['licks'].data)),axis=0)
    licks_ts_grand = np.concatenate((licks_grand,np.array(dataDict['licks'].timestamps)),axis=0)
    
    rewards_grand = np.concatenate((rewards_grand,np.array(dataDict['rewards'].data)),axis=0)
    rewards_ts_grand = np.concatenate((rewards_grand,np.array(dataDict['rewards'].timestamps)),axis=0)
    
    
dff = df_tuple(dff_ts_grand,dff_grand)
speed = df_tuple(speed_ts_grand,speed_grand)
pupil = df_tuple(pupil_ts_grand,pupil_grand)
licks = df_tuple(licks_ts_grand,licks_grand)
rewards = df_tuple(rewards_ts_grand,rewards_grand)


