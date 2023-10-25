#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:10:00 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from collections import namedtuple
import seaborn as sbn
import h5py
import numpy as np
from tqdm import tqdm
import re
import os
import socket
import models.probe
import matplotlib.pyplot as plt
import models.wav2vec2.feature_extraction_wav2vec2
from models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
import models.wav2vec2.modeling_wav2vec2
from models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTraining
import models.wav2vec2.configuration_wav2vec2
from models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

# from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import get_full_repo_name, send_example_telemetry


hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'


fname_load = os.path.join(base,'analyses/wav2vec2/datasets/dataset_probe.h5')

def load_h5dataset(fname_dset,field):
    with h5py.File(fname_dset,'r') as f:
        if field == 'dict_labels':
            dset = {}
            for key in f[field].keys():
                dset[key] = np.array(f[field][key])

        else:
            dset = []
            for i in tqdm(range(len(f[field].keys()))):
                rgb = {}
                idx= str(i)
                for key in f[field][idx].keys():
                    if key=='input_values':
                        rgb[key] = np.array(f[field][idx][key],dtype='float32')
                    else:
                        rgb[key] = np.array(f[field][idx][key])
                dset.append(rgb)
    return dset

dset_svmtrain = load_h5dataset(fname_load,'dset_svmtrain')
dset_svmtest1 = load_h5dataset(fname_load,'dset_svmtest1')
dset_svmtest2 = load_h5dataset(fname_load,'dset_svmtest2')
dict_labels = load_h5dataset(fname_load,'dict_labels')
labels_svmtrain = dict_labels['labels_svmtrain']
labels_svmtest1 = dict_labels['labels_svmtest1']
labels_svmtest2 = dict_labels['labels_svmtest2']



context_len = dset_svmtrain[0]['input_values'].shape[0]

# %% Context vectors test: Probe model
mdl_dir = os.path.join(base,'analyses/wav2vec2/','models/wav2vec2-3d-LN-spatial-LR-0.0005-contLen-32-mask-10-dropOut-0.5-hidden-0.5')

rgb = os.listdir(mdl_dir)
pattern = r'_(\d+)'
result = [int(re.search(pattern,s).group(1)) for s in rgb if 'cpt' in s]

last_cpt = max(result)
cpts_all = [0,500,last_cpt] #np.arange(0,1050,50) #[0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500]

keys_to_remove = ('mouse_id','ophys_exp_id','stiminfo')


n_loops = 20
l = 0

score1_stim_loop = np.zeros((len(cpts_all),n_loops));score1_stim_loop[:]=np.nan
score1_nonstim_loop = np.zeros((len(cpts_all),n_loops));score1_nonstim_loop[:]=np.nan
score1_total_loop = np.zeros((len(cpts_all),n_loops));score1_total_loop[:]=np.nan
score2_stim_loop = np.zeros((len(cpts_all),n_loops));score2_stim_loop[:]=np.nan
score2_nonstim_loop = np.zeros((len(cpts_all),n_loops));score2_nonstim_loop[:]=np.nan
score2_total_loop = np.zeros((len(cpts_all),n_loops));score2_total_loop[:]=np.nan
score_noiseAsStim_loop = np.zeros((len(cpts_all),n_loops));score_noiseAsStim_loop[:]=np.nan
score_noiseAsNonstim_loop = np.zeros((len(cpts_all),n_loops));score_noiseAsNonstim_loop[:]=np.nan


# temp_dim = 32


c=10
for c in range(0,len(cpts_all)):
    cpt = cpts_all[c]
    if cpt==0:
        mdl_dir_cpt = os.path.join(mdl_dir,'cpt_'+str(25)+'/')
        model = Wav2Vec2ForPreTraining.from_pretrained(mdl_dir_cpt)
        config = model.config
        model = Wav2Vec2ForPreTraining(config)
    else:
        mdl_dir_cpt = os.path.join(mdl_dir,'cpt_'+str(cpt)+'/')
        model = Wav2Vec2ForPreTraining.from_pretrained(mdl_dir_cpt)
        config = model.config
    
    
    temp_dim = int(models.probe.get_layerTempOutputDim(context_len,config.conv_kernel,config.conv_stride))
    model = model.cuda()
    
    mask_time_prob = config.mask_time_prob
    mask_time_length = config.mask_time_length

    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
    )
    
    vec_dset_svmtrain = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_svmtrain))
    dataloader_svmtrain = DataLoader(vec_dset_svmtrain, collate_fn=data_collator, batch_size=32)
    
    vec_dset_svmtest1 = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_svmtest1))
    dataloader_svmtest1 = DataLoader(vec_dset_svmtest1, collate_fn=data_collator, batch_size=32)
    
    vec_dset_svmtest2 = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_svmtest2))
    dataloader_svmtest2 = DataLoader(vec_dset_svmtest2, collate_fn=data_collator, batch_size=32)
    
    # vec_dset_noise = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_noisemovie))
    # dataloader_noise = DataLoader(vec_dset_noise, collate_fn=data_collator, batch_size=32)
    
    
    hidden_size = config.hidden_size
    l=0
    for l in range(n_loops):
    
        cv_svmtrain_all = np.zeros((0,temp_dim,hidden_size))
        model.eval()
        for step, batch in enumerate(dataloader_svmtrain):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_svmtrain_all = np.concatenate((cv_svmtrain_all,rgb),axis=0)
                
                
        cv_svmtest1_all = np.zeros((0,temp_dim,hidden_size))
        # model.eval()
        for step, batch in enumerate(dataloader_svmtest1):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_svmtest1_all = np.concatenate((cv_svmtest1_all,rgb),axis=0)
        
        
        cv_svmtest2_all = np.zeros((0,temp_dim,hidden_size))
        # model.eval()
        for step, batch in enumerate(dataloader_svmtest2):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_svmtest2_all = np.concatenate((cv_svmtest2_all,rgb),axis=0)
        
        
        idx_entry = -1
        cv_svmtrain = cv_svmtrain_all[:,idx_entry,:]
        cv_svmtest1 = cv_svmtest1_all[:,idx_entry,:]
        cv_svmtest2 = cv_svmtest2_all[:,idx_entry,:]
        
        # cv_svmtrain = cv_svmtrain.reshape(-1,cv_svmtrain.shape[1]*cv_svmtrain.shape[2])
        # cv_svmtest1 = cv_svmtest1.reshape(-1,cv_svmtest1.shape[1]*cv_svmtest1.shape[2])
        # cv_svmtest2 = cv_svmtest2.reshape(-1,cv_svmtest2.shape[1]*cv_svmtest2.shape[2])

        cv_svmtrain_norm = cv_svmtrain #(cv_svmtrain-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_svmtest1_norm = cv_svmtest1 #(cv_svmtest1-np.min(cv_svmtest1,axis=0)[None,:])/(np.max(cv_svmtest1,axis=0)[None,:]-np.min(cv_svmtest1,axis=0)[None,:])
        cv_svmtest2_norm = cv_svmtest2 #(cv_svmtest2-np.min(cv_svmtest2,axis=0)[None,:])/(np.max(cv_svmtest2,axis=0)[None,:]-np.min(cv_svmtest2,axis=0)[None,:])
        
        
        # num_cats = np.array([temp_dim]) #np.unique(np.concatenate([labels_svmtrain,labels_svmtest1,labels_svmtest2])).astype('int')
        num_cats = np.unique(np.concatenate([labels_svmtrain,labels_svmtest1,labels_svmtest2])).astype('int')
        if num_cats.shape[0] == 1: # i.e. if we are high and lows in the sequence
            labels_svmtrain_probe = np.asarray(labels_svmtrain_fullseq).astype('float64')
            labels_svmtest1_probe = np.asarray(labels_svmtest1_fullseq).astype('float64')
            labels_svmtest2_probe = np.asarray(labels_svmtest2_fullseq).astype('float64')
            
            labels_train_probe = np.asarray(labels_svmtrain_fullseq)
            labels_test1_probe = np.asarray(labels_svmtest1_fullseq)
            labels_test2_probe = np.asarray(labels_svmtest2_fullseq)
        
        else:   # catergories = number of images or image vs non image 
            labels_svmtrain_probe = np.zeros((labels_svmtrain.shape[0],len(num_cats)))     # [stim,nonstim]
            for n in num_cats:
                labels_svmtrain_probe[labels_svmtrain==n,n] = 1
    
            labels_svmtest1_probe = np.zeros((labels_svmtest1.shape[0],len(num_cats)))     # [stim,nonstim]
            for n in num_cats:
                labels_svmtest1_probe[labels_svmtest1==n,n] = 1
    
            labels_svmtest2_probe = np.zeros((labels_svmtest2.shape[0],len(num_cats)))     # [stim,nonstim]
            for n in num_cats:
                labels_svmtest2_probe[labels_svmtest2==n,n] = 1
                
            labels_train_probe = labels_svmtrain
            labels_test1_probe = labels_svmtest1
            labels_test2_probe = labels_svmtest2

    
        idx_shuffled = np.arange(0,cv_svmtrain_norm.shape[0])
        np.random.shuffle(idx_shuffled)
        
        config_probe = dict(
                dim_inpFeatures = cv_svmtrain_norm.shape[-1],
                nunits_hidden = [128,], #[32,],
                nunits_out = len(num_cats),
                lr = 1e-6
                )
        rgb=namedtuple('config_probe',config_probe)
        config_probe=rgb(**config_probe)
        
        n_epochs = 2000
        batch_size = 512
        kernel_reg = 1e-5
        
        loss_fn = nn.CrossEntropyLoss()
        

        probe_layers = [models.probe.probe_hidden(config_probe, layer_id=i) for i in range(len(config_probe.nunits_hidden)+1)]
        mdl_probe = models.probe.ProbeModel(probe_layers)
        mdl_probe = mdl_probe.cuda()
        optimizer = optim.AdamW(mdl_probe.parameters(), lr=config_probe.lr)
        params = []
       
        
        loss_train_grand = []
        loss_val_grand = []
        scoreT = []
        scoreV = []
        
        batches_per_epoch = (len(cv_svmtrain_norm)// batch_size)+1
        epoch=0
        for epoch in range(n_epochs):
            batch=0
            labels_pred = []
        
            # print('epoch: %d of %d'%(epoch+1,n_epochs))
            i=0
            for i in range(0,batches_per_epoch):
                batch+=1
                mdl_probe.train()
                
                state_dict = mdl_probe.state_dict()
                params.append(state_dict)
                
                start = i * batch_size
                X = cv_svmtrain_norm[idx_shuffled[start:start+batch_size]]
                y = labels_svmtrain_probe[idx_shuffled[start:start+batch_size]]
            
                X = torch.tensor(X).cuda()
                y = torch.tensor(y).cuda()
                    
                y_pred = mdl_probe(X)
                loss = loss_fn(y_pred, y)
                loss = loss+models.probe.l2_reg(kernel_reg,mdl_probe)     
                
                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        
                y_pred = y_pred.detach().cpu().numpy()
                if y_pred.shape[-1] == temp_dim:
                    y_pred_logit = np.round(y_pred)
                else:
                    y_pred_logit = np.argmax(y_pred,axis=1)
                labels_pred = labels_pred + y_pred_logit.tolist()
                
                
            # Training acc
            labels_pred = np.asarray(labels_pred)
            labels_svmtrain_temp = labels_train_probe[idx_shuffled]
            
            tp = np.logical_and(labels_pred>0, labels_svmtrain_temp>0)
            fp = np.logical_and(labels_pred>0, labels_svmtrain_temp==0)
            fn = np.logical_and(labels_pred==0, labels_svmtrain_temp>0)
            prec = tp.sum()/(tp.sum()+fp.sum())
            rec = tp.sum()/(tp.sum()+fn.sum())
            fscore = (2*prec*rec)/(prec+rec)
            
            scoreT_stim = (labels_pred[labels_svmtrain_temp!=0] == labels_svmtrain_temp[labels_svmtrain_temp!=0]).sum()/len(labels_svmtrain_temp[labels_svmtrain_temp!=0])
            scoreT_nonstim = (labels_pred[labels_svmtrain_temp==0] == labels_svmtrain_temp[labels_svmtrain_temp==0]).sum()/len(labels_svmtrain_temp[labels_svmtrain_temp==0])
            scoreT_total = fscore#(labels_pred == labels_svmtrain_temp).sum()/len(labels_svmtrain_temp)
            print('Epoch: %d -- Loss: %f, Test Set 1 Accuracy - Stim: %0.2f | NonStim: %0.2f | F-score: %0.2f' %(epoch,loss,scoreT_stim,scoreT_nonstim,scoreT_total))
            
            
            # Validation acc
            mdl_probe.eval()
            X = torch.tensor(cv_svmtest1_norm).cuda()
            y_pred = mdl_probe(X)
            loss_val = loss_fn(y_pred,torch.tensor(labels_svmtest1_probe).cuda())
            loss_val = loss_val+models.probe.l2_reg(kernel_reg,mdl_probe)     
        
            y_pred = y_pred.detach().cpu().numpy()
            if y_pred.shape[-1] == temp_dim:
                labels_svmtest1_pred = np.round(y_pred)
            else:
                labels_svmtest1_pred = np.argmax(y_pred,axis=1)
            
            tp = np.logical_and(labels_svmtest1_pred>0, labels_test1_probe>0)
            fp = np.logical_and(labels_svmtest1_pred>0, labels_test1_probe==0)
            fn = np.logical_and(labels_svmtest1_pred==0, labels_test1_probe>0)
            prec = tp.sum()/(tp.sum()+fp.sum())
            rec = tp.sum()/(tp.sum()+fn.sum())
            fscore = (2*prec*rec)/(prec+rec)

            scoreV_stim = (labels_svmtest1_pred[labels_test1_probe!=0] == labels_test1_probe[labels_test1_probe!=0]).sum()/len(labels_test1_probe[labels_test1_probe!=0])
            scoreV_nonstim = (labels_svmtest1_pred[labels_test1_probe==0] == labels_test1_probe[labels_test1_probe==0]).sum()/len(labels_test1_probe[labels_test1_probe==0])
            scoreV_total = fscore #(labels_svmtest1_pred == labels_test1_probe).sum()/len(labels_test1_probe)
            print('Validation -- Loss: %f, Test Set 1 Accuracy - Stim: %0.2f | NonStim: %0.2f | F-score: %0.2f' %(loss_val,scoreV_stim,scoreV_nonstim,scoreV_total))
        
        
            loss_train_grand.append(loss.detach().cpu().numpy())
            loss_val_grand.append(loss_val.detach().cpu().numpy())
            scoreT.append((scoreT_stim,scoreT_nonstim,scoreT_total))
            scoreV.append((scoreV_stim,scoreV_nonstim,scoreV_total))
            
        
        scoreT = np.asarray(scoreT)
        scoreV = np.asarray(scoreV)
        
        
        fig,axs = plt.subplots(2,2,figsize=(20,10))  
        axs = np.ravel(axs)
        axs[0].plot(loss_train_grand,label='train');axs[0].plot(loss_val_grand,label='val');axs[0].legend();axs[0].set_title('loss')
        axs[1].plot(scoreT[:,2],label='train');axs[1].plot(scoreV[:,2],label='val');axs[1].legend();axs[1].set_title('Total')
        axs[2].plot(scoreT[:,0],label='train');axs[2].plot(scoreV[:,0],label='val');axs[2].legend();axs[2].set_title('Stim')
        axs[3].plot(scoreT[:,1],label='train');axs[3].plot(scoreV[:,1],label='val');axs[3].legend();axs[3].set_title('Nonstim')
        plt.show()
        
        # Evaluate model on held out data
        bestEpoch = np.argmax(scoreV[:,-1])
        print('best epoch = %d'%bestEpoch)
        mdl_probe.load_state_dict(params[bestEpoch])
        
        
        mdl_probe.eval()
        
        # labels_svmtest1_probe = np.zeros((labels_svmtest1.shape[0],config_probe.nunits_out))     # [stim,nonstim]
        # labels_svmtest1_probe[labels_svmtest1==0,0] = 1
        # labels_svmtest1_probe[labels_svmtest1==1,1] = 1
        X = torch.tensor(cv_svmtest1_norm).cuda()
        y_pred = mdl_probe(X).detach().cpu().numpy()
        if y_pred.shape[-1] == temp_dim:
            labels_svmtest1_pred = np.round(y_pred)
        else:
            labels_svmtest1_pred = np.argmax(y_pred,axis=1)
        
        
        # labels_svmtest2_probe = np.zeros((labels_svmtest2.shape[0],config_probe.nunits_out))     # [stim,nonstim]
        # labels_svmtest2_probe[labels_svmtest2==0,0] = 1
        # labels_svmtest2_probe[labels_svmtest2==1,1] = 1
        X = torch.tensor(cv_svmtest2_norm).cuda()
        y_pred = mdl_probe(X).detach().cpu().numpy()
        if y_pred.shape[-1] == temp_dim:
            labels_svmtest2_pred = np.round(y_pred)
        else:
            labels_svmtest2_pred = np.argmax(y_pred,axis=1)

        
        
        tp = np.logical_and(labels_svmtest1_pred>0, labels_test1_probe>0)
        fp = np.logical_and(labels_svmtest1_pred>0, labels_test1_probe==0)
        fn = np.logical_and(labels_svmtest1_pred==0, labels_test1_probe>0)
        prec = tp.sum()/(tp.sum()+fp.sum())
        rec = tp.sum()/(tp.sum()+fn.sum())
        fscore = (2*prec*rec)/(prec+rec)

        score1_stim = (labels_svmtest1_pred[labels_test1_probe!=0] == labels_test1_probe[labels_test1_probe!=0]).sum()/len(labels_test1_probe[labels_test1_probe!=0])
        score1_nonstim = (labels_svmtest1_pred[labels_test1_probe==0] == labels_test1_probe[labels_test1_probe==0]).sum()/len(labels_test1_probe[labels_test1_probe==0])
        score1_total = fscore #(labels_svmtest1_pred == labels_svmtest1).sum()/len(labels_svmtest1)
        print('Test Set 1 Accuracy - Stim: %0.2f | NonStim: %0.2f | F-score: %0.2f' %(score1_stim,score1_nonstim,score1_total))
        
        
        tp = np.logical_and(labels_svmtest2_pred>0, labels_test2_probe>0)
        fp = np.logical_and(labels_svmtest2_pred>0, labels_test2_probe==0)
        fn = np.logical_and(labels_svmtest2_pred==0, labels_test2_probe>0)
        prec = tp.sum()/(tp.sum()+fp.sum())
        rec = tp.sum()/(tp.sum()+fn.sum())
        fscore = (2*prec*rec)/(prec+rec)

        score2_stim = (labels_svmtest2_pred[labels_test2_probe!=0] == labels_test2_probe[labels_test2_probe!=0]).sum()/len(labels_test2_probe[labels_test2_probe!=0])
        score2_nonstim = (labels_svmtest2_pred[labels_test2_probe==0] == labels_test2_probe[labels_test2_probe==0]).sum()/len(labels_test2_probe[labels_test2_probe==0])
        score2_total = fscore# (labels_svmtest2_pred == labels_svmtest2).sum()/len(labels_svmtest2)
        print('Test Set 2 Accuracy - Stim: %0.2f | NonStim: %0.2f | F-score: %0.2f' %(score2_stim,score2_nonstim,score2_total))
        
        score1_stim_loop[c,l] = score1_stim
        score1_nonstim_loop[c,l] = score1_nonstim
        score1_total_loop[c,l] = score1_total
        score2_stim_loop[c,l] = score2_stim
        score2_nonstim_loop[c,l] = score2_nonstim
        score2_total_loop[c,l] = score2_total
        # score_noiseAsStim_loop[c,l] = score2_noise_stim
        # score_noiseAsNonstim_loop[c,l] = score2_noise_nonstim

score1_stim_avg = np.nanmean(score1_stim_loop,axis=-1)
score1_nonstim_avg = np.nanmean(score1_nonstim_loop,axis=-1)
score1_total_avg = np.nanmean(score1_total_loop,axis=-1)
score1_acc_avg = (score1_stim_avg+score1_nonstim_avg)/2

score2_stim_avg = np.nanmean(score2_stim_loop,axis=-1)
score2_nonstim_avg = np.nanmean(score2_nonstim_loop,axis=-1)
score2_total_avg = np.nanmean(score2_total_loop,axis=-1)
score2_acc_avg = (score2_stim_avg+score2_nonstim_avg)/2

# % Plot f-score
labels_scores = cpts_all
x_inter = 1
if x_inter>1:
    y_range = (0.2,0.9)
    fig,axs=plt.subplots(1,2,figsize=(20,5));axs=np.ravel(axs)
    axs[0].plot(score1_total_avg,label='set 1')
    axs[0].plot(score2_total_avg,label='set 2')
    axs[0].set_yticks(np.arange(0,1.1,.1))
    axs[0].set_ylim(y_range)
    axs[0].set_xticks(np.arange(0,len(cpts_all),x_inter))
    axs[0].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter,'int')])
    axs[0].grid(axis='y',color=[.7,.7,.7],linestyle='--')
    axs[0].set_title('fscore')
    axs[0].legend()
    
    axs[1].plot(score1_acc_avg,label='set 1')
    axs[1].plot(score2_acc_avg,label='set 2')
    axs[1].set_yticks(np.arange(0,1.1,.1))
    axs[1].set_ylim(y_range)
    axs[1].set_xticks(np.arange(0,len(cpts_all),x_inter))
    axs[1].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter,'int')])
    axs[1].grid(axis='y',color=[.7,.7,.7],linestyle='--')
    axs[1].set_title('Accuracy')
    axs[1].legend()
else:
    
    y_range = (0.3,0.8)
    fig,axs=plt.subplots(1,2,figsize=(8,3));axs=np.ravel(axs)
    axs[0].boxplot(score1_total_loop.T,labels=cpts_all)
    axs[0].set_yticks(np.arange(0,1.1,.1))
    axs[0].set_ylim(y_range)
    axs[0].grid(axis='y',color=[.7,.7,.7],linestyle='--')
    axs[0].set_title('fscore - Set 1')
    axs[0].set_ylabel('F-score')
    axs[0].set_xlabel('Epoch')

    axs[1].boxplot(score2_total_loop.T,labels=cpts_all)
    axs[1].set_yticks(np.arange(0,1.1,.1))
    axs[1].set_ylim(y_range)
    axs[1].grid(axis='y',color=[.7,.7,.7],linestyle='--')
    axs[1].set_title('fscore - Set 2')
    axs[1].set_ylabel('F-score')
    axs[1].set_xlabel('Epoch')



# %%

labels_scores = cpts_all
x_inter = 4
y_range = (0.3,0.9)
fig,axs=plt.subplots(1,2,figsize=(20,5));axs=np.ravel(axs)
axs[0].plot((score1_stim_avg+score1_nonstim_avg)/2,label='total')
axs[0].plot(score1_stim_avg,label='stim')
axs[0].plot(score1_nonstim_avg,label='non-stim')
# axs[0].set_yticks(np.arange(0,1.1,.1))
# axs[0].set_ylim(y_range)
axs[0].set_xticks(np.arange(0,len(cpts_all),x_inter))
axs[0].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter,'int')])
axs[0].grid(axis='y',color=[.7,.7,.7],linestyle='--')
axs[0].set_title('Set 1')
axs[0].legend()

axs[1].plot((score2_stim_avg+score2_nonstim_avg)/2,label='total')
axs[1].plot(score2_stim_avg,label='stim')
axs[1].plot(score2_nonstim_avg,label='non-stim')
# axs[1].set_yticks(np.arange(0,1.1,.1))
# axs[1].set_ylim(y_range)
axs[1].set_xticks(np.arange(0,len(cpts_all),x_inter))
axs[1].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter,'int')])
axs[1].grid(axis='y',color=[.7,.7,.7],linestyle='--')
axs[1].set_title('Set 2')
axs[1].legend()


# axs[1].plot(score2_stim_loop);axs[1].set_ylim([0,1.1])
# axs[1].set_yticks(np.arange(0,1.1,.1))
# axs[1].set_ylim(y_range)
# axs[1].grid(axis='y',color=[.7,.7,.7],linestyle='--')
# axs[1].set_xticks(np.arange(0,len(cpts_all),x_inter))
# axs[1].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter)])
# axs[1].set_title('stim induced')


# axs[2].plot(score2_nonstim_loop);axs[2].set_ylim([0,1.1])
# axs[2].set_yticks(np.arange(0,1.1,.1))
# axs[2].set_ylim(y_range)
# axs[2].grid(axis='y',color=[.7,.7,.7],linestyle='--')
# axs[2].set_xticks(np.arange(0,len(cpts_all),x_inter))
# axs[2].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter)])
# axs[2].set_title('non-stim induced')


