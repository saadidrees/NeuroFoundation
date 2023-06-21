#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:25:03 2023

@author: Saad Idrees, jZ Lab
"""

"""
- stimulus_presentations is the experimental protocol
- cell_specimen_table: info about identified/isolated cells
- dff_traces number of rows correspond to number of cells. So it should have same num of rows as cell_specimen_table
- events: Contains an array same size as dff for each cell. Apparently contains spiking events derived or detected from 2-photon movies
- licks: timestamps of licks in seconds (columns 1) and in frames (column 2). Sampled @ 60 Hz.
- rewards: [volume,timestamps,autor_rewarded]
q. Whats the diff between cell speciment id and roi id?
"""

import numpy as np
import pandas as pd
global dataset
import seaborn as sns
from collections import namedtuple
df_tuple = namedtuple('df_tuple', ['timestamps', 'data'])




def get_dff(cell_specimen_id,initial_time,final_time):
    dff = ds.dff_traces.loc[cell_specimen_id].dff
    timestamps = ds.ophys_timestamps
    
    df = {'data': dff,'timestamps': timestamps}
    df = pd.DataFrame(df)
    df = df.query('timestamps >= @initial_time and timestamps < @final_time')
    
    return df,df.data,df.timestamps


def get_events(cell_specimen_id,initial_time,final_time):
    events = ds.events.loc[cell_specimen_id].events
    timestamps = ds.ophys_timestamps
    
    df = {'data': events,'timestamps': timestamps}
    df = pd.DataFrame(df)
    df = df.query('timestamps >= @initial_time and timestamps < @final_time')
    
    return df,df.data,df.timestamps


def get_stim_pres(stimulus_presentations,initial_time,final_time):
    stim_pres_chunk = stimulus_presentations.query('end_time >= @initial_time and start_time <= @final_time')
    return stim_pres_chunk
    

def get_stim_frame(dataset,stimulus_presentations,stim_pres_time,timeunit='s'):
    if timeunit=='s':   # seconds
        stim_frame = stimulus_presentations.query('end_time >= @stim_pres_time')
    elif timeunit=='f': # frames
        stim_frame = stimulus_presentations.query('end_frame >= @stim_pres_time')
    stim_info = stim_frame.iloc[0]
    
    stim_img = dataset.stimulus_templates.loc[stim_info.image_name].warped
    
    return stim_img,stim_info


def get_speed(initial_time,final_time):
    speed = ds.running_speed.speed
    timestamps = ds.running_speed.timestamps
    
    df = {'data': speed,'timestamps': timestamps}
    df = pd.DataFrame(df)
    df = df.query('timestamps >= @initial_time and timestamps < @final_time')
    
    return df

def get_pupil(initial_time,final_time):
    pupil = ds.eye_tracking.pupil_width
    timestamps = ds.eye_tracking.timestamps
    df = {'data': pupil,'timestamps': timestamps}
    df = pd.DataFrame(df)
    df = df.query('timestamps >= @initial_time and timestamps < @final_time')
    return df

def get_licks(initial_time,final_time):
    timestamps = ds.licks.timestamps
    licks = np.zeros_like(timestamps)
    df = {'data': licks,'timestamps': timestamps}
    df = pd.DataFrame(df)
    df = df.query('timestamps >= @initial_time and timestamps < @final_time')
    return df

def get_rewards(initial_time,final_time):
    timestamps = ds.rewards.timestamps
    rewards = np.zeros_like(timestamps)
    df = {'data': rewards,'timestamps': timestamps}
    df = pd.DataFrame(df)
    df = df.query('timestamps >= @initial_time and timestamps < @final_time')
    return df

def get_dataDict(dataset,cell_specimen_id,initial_time,final_time):
    global ds
    ds = dataset
    unique_stimuli = [stimulus for stimulus in dataset.stimulus_presentations['image_name'].unique()]
    colormap = {image_name: sns.color_palette()[image_number] for image_number,image_name in enumerate(np.sort(unique_stimuli))}
    colormap['omitted'] = (1,1,1) # set omitted stimulus to white color
    stimulus_presentations = dataset.stimulus_presentations
    stimulus_presentations['color'] = dataset.stimulus_presentations['image_name'].map(lambda image_name: colormap[image_name])
    
    cell_id = cell_specimen_id[0]
    df,dff_rgb,dff_timestamps = get_dff(cell_id,initial_time,final_time)
    df_events,events_rgb,events_timestamps = get_events(cell_id,initial_time,final_time)
    
    dff_data = dff_rgb[:,None]
    events_data = events_rgb[:,None]
    for cell_id in cell_specimen_id:
        _,dff_rgb,_ = get_dff(cell_id,initial_time,final_time)
        dff_data = np.hstack((dff_data,dff_rgb[:,None]))
        
        _,events_rgb,_ = get_events(cell_id,initial_time,final_time)
        events_data = np.hstack((events_data,events_rgb[:,None]))
        
    dff = df_tuple(dff_timestamps,dff_data)
    events = df_tuple(events_timestamps,events_data)
    
    stim_pres = get_stim_pres(stimulus_presentations,initial_time,final_time)
    
    
    speed = get_speed(initial_time,final_time)
    pupil = get_pupil(initial_time,final_time);
    pupil.data=(pupil.data-pupil.data.mean());pupil.data=pupil.data+np.abs(pupil.data.min())
    licks = get_licks(initial_time,final_time)
    rewards = get_rewards(initial_time,final_time)
    
    dataDict = dict(
                    dff=dff,
                    events=events,
                    stim_pres=stim_pres,
                    stimulus_presentations=stimulus_presentations,
                    speed=speed,
                    pupil=pupil,
                    licks=licks,
                    rewards=rewards)
    
    return dataDict
    
    
