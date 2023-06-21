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


# %% Select Experiment to download
ophys_exp_id = 957759570
dataset = cache.get_behavior_ophys_experiment(ophys_exp_id)

# %% Extract and plot data
cell_specimen_table = dataset.cell_specimen_table
cell_specimen_ids = cell_specimen_table.index.values

initial_time = 820
final_time = 860
idx_cell = np.arange(0,5)
cell_specimen_id = cell_specimen_ids[idx_cell]

dataDict = dataExtractor.get_dataDict(dataset,cell_specimen_id,initial_time,final_time)

dff = dataDict['dff']
events = dataDict['events']
stim_pres = dataDict['stim_pres']
stimulus_presentations = dataDict['stimulus_presentations']
speed = dataDict['speed']
pupil = dataDict['pupil']
licks = dataDict['licks']
rewards = dataDict['rewards']


fig,axs = plt.subplots(2,1,figsize=(15,7))
axs = np.ravel(axs)

axs[0].plot(dff.timestamps,dff.data,label='dF/F')
axs[0].plot(events.timestamps,events.data,label='events')
for _,stimulus in stim_pres.iterrows():
    axs[0].axvspan(stimulus['start_time'],stimulus['end_time'],color=stimulus['color'],alpha=.25)
axs[0].legend()

axs[1].plot(speed.timestamps,speed.data,label='speed')
axs[1].plot(pupil.timestamps,pupil.data,label='pupil')
axs[1].plot(licks.timestamps,licks.data,'o',markersize=3,color='black',label='licks')
axs[1].plot(rewards.timestamps,rewards.data,marker='d',markersize=12,color='green',label='rewards',alpha=0.5)
axs[1].legend()

# %%
timeForStimImgExtract = 80000;timeunit='f'
stim_img,stim_info = dataExtractor.get_stim_frame(dataset,stimulus_presentations,timeForStimImgExtract,timeunit=timeunit)

fig,axs = plt.subplots(1,1)
axs = np.ravel(axs)
axs[0].imshow(stim_img,cmap='gray')
axs[0].set_title('time: %d %s'%(timeForStimImgExtract,timeunit))
