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


# %% Select 
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
"""
# Define selecting params
select_targetStructure = 'VISam'
n_mice = 3
n_cells = -1
initial_time = 100
n_dur = 2000
thresh_numSessions = 0
bin_ms = 16 


final_time = initial_time + n_dur

# Initialize empty variables where the number of rows will eventually correspond to number of cells
dff_grand = 0
speed_grand = 0
pupil_grand = 0
ts_rs_grand = 0
imgName_rs_grand = 0
total_cells = 0


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
select_container = containerIds_nmice[3]
for select_container in containerIds_nmice:
    
    #4
    exp_table_subset = exp_table_subArea.query('ophys_container_id == @select_container')# and targeted_structure == @select_targetStructure')
    ophys_exp_ids_inContainer = exp_table_subset.index.values
    
    
    #5 loop over experiments in container and get list of intersection cells
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
    idx_cells = cell_id_counts>thresh_numSessions
    cell_ids_full = cell_id_unique[idx_cells]
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
        df = pd.DataFrame({'cell_id':cell_ids_full[i],'exp_id':rgb})
        exp_ids_perCell = pd.concat((exp_ids_perCell,df),axis=0)
            
    # % Resample Stimulus and build stimulus array 
    dff_cont = np.zeros((cell_ids_full.shape[0]),dtype='object')
    speed_cont = np.zeros((cell_ids_full.shape[0]),dtype='object')
    pupil_cont = np.zeros((cell_ids_full.shape[0]),dtype='object')
    ts_rs_cont = np.zeros((cell_ids_full.shape[0]),dtype='object')
    imgNames_rs_cont = np.zeros((cell_ids_full.shape[0]),dtype='object')
    
    
    # Inner loop over experiments. This should append 'sessions'
    ophys_exp_id = ophys_exp_ids_inContainer[0]
    for ophys_exp_id in ophys_exp_ids_inContainer:
        dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)
        
        
        cell_id_toExtract = np.array(exp_ids_perCell.query('exp_id == @ophys_exp_id').cell_id)
        cell_idxInFull = np.intersect1d(cell_id_toExtract,cell_ids_full,return_indices=True)[2]
        
        dataDict = dataExtractor.get_dataDict(dataset,cell_id_toExtract,initial_time,final_time)
        stim_pres_orig = dataDict['stim_pres']
        stimulus_presentations_orig = dataDict['stimulus_presentations']
    
        stim_startTimes = np.round(np.array(stim_pres_orig.start_time)*1000)
        stim_endTimes = np.round(np.array(stim_pres_orig.end_time)*1000)
        stim_times = np.concatenate((stim_startTimes[:,None],stim_endTimes[:,None]),axis=1).astype('int64')
        
        ts_1ms = np.arange(stim_startTimes[0],stim_endTimes[-1])
        ts_rs = ts_1ms[::bin_ms]
        
        
        Y_1ms = np.zeros(ts_1ms.shape,dtype='object')
        i=0
        for i in range(stim_times.shape[0]):
            img_name = stim_pres_orig.iloc[i].image_name
            idx_range = (np.arange(stim_times[i,0],stim_times[i,1])-ts_1ms[0]).astype('int32')
            Y_1ms[idx_range] = img_name
        
        
        Y_rs = Y_1ms[::bin_ms]
        
    
        initial_time_fromStim = np.floor((ts_rs[0]/1000)-1).astype('int')
        final_time_fromStim = np.ceil((ts_rs[-1]/1000)+1).astype('int')
    
        dataDict = dataExtractor.get_dataDict(dataset,cell_id_toExtract,initial_time_fromStim,final_time_fromStim)
        dff_orig = dataDict['dff']
        events_orig = dataDict['events']
        speed_orig = dataDict['speed']
        pupil_orig = dataDict['pupil']
        licks_orig = dataDict['licks'];licks_orig.data = 1
        rewards_orig = dataDict['rewards'];rewards_orig.data = 1
    
        dff = dataExtractor.resample(dff_orig,ts_rs,initial_time_fromStim,final_time_fromStim)
        speed = dataExtractor.resample(speed_orig,ts_rs,initial_time,final_time)
        pupil = dataExtractor.resample(pupil_orig,ts_rs,initial_time,final_time)
        licks = dataExtractor.resample(licks_orig,ts_rs,initial_time,final_time,interp=False)
        rewards = dataExtractor.resample(rewards_orig,ts_rs,initial_time,final_time)
        
        for i in range(cell_idxInFull.shape[0]):
            if np.isscalar(dff_cont[cell_idxInFull[i]]):
                dff_cont[cell_idxInFull[i]] = [dff[:,i]]
                speed_cont[cell_idxInFull[i]] = [speed]
                pupil_cont[cell_idxInFull[i]] = [pupil]
                imgNames_rs_cont[cell_idxInFull[i]] = [Y_rs]
                ts_rs_cont[cell_idxInFull[i]] = [ts_rs]
            else:
                dff_cont[cell_idxInFull[i]].append(dff[:,i])
                speed_cont[cell_idxInFull[i]].append(speed)
                pupil_cont[cell_idxInFull[i]].append(pupil)
                imgNames_rs_cont[cell_idxInFull[i]].append(Y_rs)
                ts_rs_cont[cell_idxInFull[i]].append(ts_rs)
    
    
    if np.isscalar(dff_grand):
        dff_grand = dff_cont
        speed_grand = speed_cont
        pupil_grand = pupil_cont
        imgNames_grand = imgNames_rs_cont
        ts_grand = ts_rs_cont
    
    else:
        dff_grand = np.concatenate([dff_grand,dff_cont],axis=0)                     # [[[cells]]][[sessions]][data]
        speed_grand = np.concatenate([speed_grand,speed_cont],axis=0)               # [[[cells]]][[sessions]][data]
        pupil_grand = np.concatenate([pupil_grand,pupil_cont],axis=0)               # [[[cells]]][[sessions]][data]
        imgNames_grand = np.concatenate([imgNames_grand,imgNames_rs_cont],axis=0)   # [[[cells]]][[sessions]][data]
        ts_grand = np.concatenate([ts_grand,ts_rs_cont],axis=0)                     # [[[cells]]][[sessions]][data]

