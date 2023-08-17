#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:27:52 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""




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


# %%
from data_handler import utils

downscale_fac = 4
chunk_size = 50

dset = []

i=0
for i in range(len(dff_mov_grand)):
    rgb = dff_mov_grand[i][0]
    rgb = rgb[::downscale_fac,::downscale_fac,:]
    rgb = np.moveaxis(rgb,-1,0)
    
    dict_metadata = dict(mouse_id=mouse_id_grand[i],
                         ophys_exp_id=ophys_exp_id_grand[i])
    temp = utils.chunker(rgb,chunk_size,dict_metadata=None)
    dset = dset+temp

dset_train = dset[:-5]
dset_val = dset[-5:]

# keys_to_remove = ('mouse_id','ophys_exp_id')
# vec_dset_train = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_train))

train_dataloader = torch.utils.data.DataLoader(
                    dset_train,
                    shuffle=False,
                    batch_size=32,
)

# %% 3D conv
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from collections import namedtuple

class fe_cnn3d(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        
        def _get_layerOutputDim(layer_id):
            if layer_id>0:
                def _conv_out_length(input_length, kernel_size, stride):
                    return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1           
                def _maxpool_out_length(input_length, kernel_size, stride):
                    return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
    
                dim_layerOut = config.dim_inputSpat
                dim_temporal = config.dim_inputTemp
                for i in range(layer_id):
                    dim_conv_out = _conv_out_length(dim_layerOut,config.conv_kernel_spatial[i],config.conv_stride_spatial[i])
                    dim_maxpool_out = _maxpool_out_length(dim_conv_out,config.maxpool_kernel_spatial[i],config.maxpool_stride_spatial[i])
                    dim_layerOut = dim_maxpool_out
                    
                    dim_temporal = _conv_out_length(dim_temporal,config.conv_kernel[i],config.conv_stride[i])
            else:
                dim_layerOut = config.dim_inputSpat

            return dim_layerOut

        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv3d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=(config.conv_kernel[layer_id],config.conv_kernel_spatial[layer_id],config.conv_kernel_spatial[layer_id]),
            stride=(config.conv_stride[layer_id],config.conv_stride_spatial[layer_id],config.conv_stride_spatial[layer_id]),
            bias=config.conv_bias,
        )
        
        self.maxpool = nn.MaxPool3d((1,config.maxpool_kernel_spatial[layer_id],config.maxpool_kernel_spatial[layer_id]),
                                    stride=(1,config.maxpool_stride_spatial[layer_id],config.maxpool_stride_spatial[layer_id]))
        dim_cnnMpOut = _get_layerOutputDim(layer_id+1)
        self.layer_norm = nn.LayerNorm([self.out_conv_dim,dim_cnnMpOut,dim_cnnMpOut],elementwise_affine=True)
        self.activation = nn.GELU()      
        
        if layer_id == len(config.conv_dim):
            assert dim_cnnMpOut==1, 'Spatial dimension not 1 in CNN output'

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.maxpool(hidden_states)
        hidden_states = hidden_states.transpose(1,2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1,2)
        hidden_states = self.activation(hidden_states)
        return hidden_states
    
    
class fe_fc(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        def _get_tempOutputDim(layer_id):
            if layer_id>0:
                def _conv_out_length(input_length, kernel_size, stride):
                    return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1           

                dim_temporal = config.dim_inputTemp
                for i in range(layer_id):                   
                    dim_temporal = _conv_out_length(dim_temporal,config.conv_kernel[i],config.conv_stride[i])
            else:
                dim_temporal = config.dim_inputTemp

            return dim_temporal

        dim_temporal =  _get_tempOutputDim(len(config.conv_dim))
        # print(dim_temporal)
        self.flatten = nn.Flatten(start_dim=1, end_dim=- 1)
        self.fc = nn.Linear(config.conv_dim[-1]*dim_temporal,config.dim_inputSpat*config.dim_inputSpat)
        self.unflatten = nn.Unflatten(0,unflattened_size=(config.dim_inputSpat,config.dim_inputSpat))
        self.layer_norm = nn.LayerNorm([config.dim_inputSpat*config.dim_inputSpat])
        self.activation = nn.Softplus()
        
        
    def forward(self, hidden_states):
        hidden_states = self.flatten(hidden_states).double()
        hidden_states = self.fc(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class mdl_comb(nn.Module):
    def __init__(self,conv_layers,fc_layer):
        super().__init__()
        self.conv_layers = nn.ModuleList(conv_layers)
        self.fc_layer = fc_layer
        
        
    def forward(self,hidden_states):
        # print(hidden_states.shape)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states).double()
        hidden_states = self.fc_layer(hidden_states)

        return hidden_states



def l2_reg(kernel_reg,mdl):
    l2_pen = 0
    for name, parameter in mdl.named_parameters():
        p_regex = re.compile(r'conv.weight')
        rgb = p_regex.search(name)
        if rgb != None:
            l2_pen = l2_pen + (kernel_reg*(parameter**2).sum())
            
    return l2_pen



def run_model(config,batch_size,n_epochs,param_counter=1):
    num_feat_extract_layers = len(config.conv_dim)
    conv_layers = [fe_cnn3d(config, layer_id=i).double() for i in range(num_feat_extract_layers)]
    fc_layer = fe_fc(config).double()
    mdl = mdl_comb(conv_layers,fc_layer)
    mdl = mdl.cuda()
    
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn=torch.nn.NLLLoss()
    optimizer = optim.Adam(mdl.parameters(), lr=config.lr,weight_decay=config.weight_decay)
    batches_per_epoch = len(dset_train)// batch_size
    epoch=0
    i=0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    
    X_val = dset_val[-1]
    y_val = X_val[-1]
    y_val = y_val[None,:,:]
    X_val = X_val[None,None,:-1]
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    for epoch in range(n_epochs):
        batch=0

        print('epoch: %d of %d'%(epoch+1,n_epochs))
        for i in range(0,batches_per_epoch):
            t_start = time.time()
            batch+=1
            if batch>50:
                break
            mdl.train()
            start = i * batch_size
            rgb = dset_train[start:start+batch_size]
            X = np.zeros((0,*rgb[0].shape))
            j=0
            for j in range(len(rgb)):
                temp = rgb[j][np.newaxis]
                X = np.concatenate((X,temp),axis=0)
            
            y = X[:,-1]
            X = X[:,:-1]
            
            X = X[:,np.newaxis]     # input channels = 1
        
            X = torch.tensor(X)
            y = torch.tensor(y)
        
            X = X.cuda()
            y = y.cuda()
            
            y_pred = mdl(X)
            y_pred = torch.unflatten(y_pred,1,(128,128))
            
            loss = loss_fn(y_pred, y)
            loss = loss+l2_reg(kernel_reg,mdl)
            idx_valid = y!=0
            idx_valid = idx_valid.cuda()
            
            
            rgb = (y_pred[idx_valid].round() - y[idx_valid]).float().detach().cpu().numpy()
            acc = np.nanmean(rgb**2)
            # store metrics
            train_loss.append(float(loss))
            train_acc.append(float(acc))
            
            # Validation
            mdl.eval()
            y_pred_val = mdl(X_val.cuda())
            y_pred_val = torch.reshape(y_pred_val,(y_pred_val.shape[0],128,128))
            v_loss = loss_fn(y_pred_val, y_val.cuda())
            idx_valid = y_val!=0
            idx_valid = idx_valid.cuda()
            
            rgb = (y_pred_val[idx_valid].round() - y_val.cuda()[idx_valid]).float().detach().cpu().numpy()
            v_acc = np.nanmean(rgb**2)

            val_loss.append(float(v_loss))
            val_acc.append(float(v_acc))
            
            # plt.imshow(y_pred_val[0,:,:].detach().cpu().numpy());plt.show()
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            
            t_elapsed = time.time()-t_start
            
            print('param_counter: %d | batch: %d | train_loss: %f | val_loss: %f | t: %d seconds'%(param_counter,batch,loss,v_loss,t_elapsed))
            
    results = dict(config=config,y_pred_val=y_pred_val.detach().cpu().numpy(),train_loss=train_loss,train_acc=train_acc,val_loss=val_loss,val_acc=val_acc)

    return results

# %% Run model
import os
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['max_split_size_mb'] = '512'

"""
chan_list = [(32,32,32,128),(16,16,64,128),(16,32,64,128),(32,32,32,512),(16,16,16,128),(16,16,16,512)]
conv_temp_list = [(3,3,2,1),(3,3,1,1),(3,2,1,1),(3,3,3,3),(5,1,1,1),(5,1,1,1)]

conv_spat_list = [
    [(3,3,3,2),(2,2,2,1),(3,2,1,1)],
    [(7,5,5,2),(2,2,1,1),(2,2,1,1)],
    [(3,2,1,1),(2,2,2,1),(2,2,2,2)],
    ]
    
"""

results_all = []
chan_list = [(16,16,16,128)]
conv_temp_list = [(5,3,1,1),(3,3,1,1),(5,5,1,1)]

conv_spat_list = [[(3,3,3,2),(2,2,2,1),(3,2,1,1)]]

kernel_reg = 1e-3

n_epochs=1
batch_size = 32

i=0;j=0;k=0;

total_params = len(chan_list)*len(conv_spat_list)*len(conv_temp_list)
param_counter=0
for i in range(len(chan_list)):
    for j in range(len(conv_temp_list)):
        for k in range(len(conv_spat_list)):
            print('i: %d | j: %d k: %d'%(i,j,k))
            param_counter+=1
            try:
                configDict = dict(
                        dim_inputSpat = 128,
                        dim_inputTemp = 49,
                        conv_dim=chan_list[i], #(32,32,32,128), 
                        conv_stride=(1,1,1,1,1), #(1,1,1,1)
                        conv_kernel=conv_temp_list[j], #(3,3,1,1),
                        conv_kernel_spatial = conv_spat_list[k][0], #(3,3,3,3),
                        conv_stride_spatial = conv_spat_list[k][1], #(3,2,1,1),
                        maxpool_kernel_spatial = conv_spat_list[k][2], #(2,2,1,1),
                        maxpool_stride_spatial = conv_spat_list[k][2], #(2,2,1,1),
                        conv_bias=True,
                        lr = 0.0001,
                        weight_decay = 0.0001,
                        )
                config=namedtuple('args',configDict)
                config=config(**configDict)
                
                results = run_model(config,batch_size,n_epochs,param_counter=param_counter)
                results_all.append(results)
                torch.cuda.empty_cache()
            except:
                pass


# %% Unfold results

train_acc_all = []
train_loss_all = []
val_acc_all = []
val_loss_all = []
y_pred_val_all = []


train_acc_all = np.array(list(map(lambda d: d['train_acc'],results_all)))
train_loss_all = np.array(list(map(lambda d: d['train_loss'],results_all)))
val_acc_all = np.array(list(map(lambda d: d['val_acc'],results_all)))
val_loss_all = np.array(list(map(lambda d: d['val_loss'],results_all)))
y_pred_val_all = np.squeeze(np.array(list(map(lambda d: d['y_pred_val'],results_all))))

plt.plot(val_loss_all[:,:].T)


idx_best_acc = np.argsort(val_acc_all[:,1])
plt.plot(val_acc_all[:,1])

plt.plot(val_acc_all[idx_best_acc[:3],:15].T)

best_config = results_all[idx_best_acc[0]]['config']

plt.imshow(y_pred_val_all[27])

"""
import h5py
fname_save = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/NeuroFoundation/cnn_hyperparams2.h5'
with h5py.File(fname_save,'w') as f:
    f.create_dataset('train_acc_all',data = train_acc_all)
    f.create_dataset('train_loss_all',data = train_loss_all)
    f.create_dataset('val_acc_all',data = val_acc_all)
    f.create_dataset('val_loss_all',data = val_loss_all)
    f.create_dataset('y_pred_val_all',data = y_pred_val_all)
    
"""

# %% CNN shape
def _get_layerOutputDim(layer_id,config):
    if layer_id>0:
        def _conv_out_length(input_length, kernel_size, stride):
            rgb = np.floor((input_length - kernel_size)/stride)+1
            return rgb
        
        def _maxpool_out_length(input_length, kernel_size, stride):
            rgb = np.floor((input_length - kernel_size)/stride)+1
            return rgb

        dim_layerOut = config.dim_inputSpat
        dim_temporal = config.dim_inputTemp
        for i in range(layer_id):
            dim_conv_out = _conv_out_length(dim_layerOut,config.conv_kernel_spatial[i],config.conv_stride_spatial[i])
            dim_maxpool_out = _maxpool_out_length(dim_conv_out,config.maxpool_kernel_spatial[i],config.maxpool_stride_spatial[i])
            dim_layerOut = dim_maxpool_out
            
            dim_temporal = _conv_out_length(dim_temporal,config.conv_kernel[i],config.conv_stride[i])
    else:
        dim_layerOut = config.dim_inputSpat
        dim_temporal = config.dim_inputTemp

    return dim_layerOut,dim_temporal

"""
conv_spat_list = [
    [(3,3,3,2),(2,2,2,1),(3,2,1,1)],
    [(7,5,5,2),(2,2,1,1),(2,2,1,1)],
    [(3,2,1,1),(2,2,2,1),(2,2,2,2)]
    ]

(7,7,5,3),(3,3,3,1),(1,1,1,1)

"""  

configDict = dict(
        dim_inputSpat = 128,
        dim_inputTemp = 49,
        conv_dim=(32,32,32,32), #(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(1,1,1,1),#(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(3,3,3,3), #(10, 3, 3, 3, 3, 2, 2),
        conv_kernel_spatial = (7,7,5,3),
        conv_stride_spatial = (3,3,3,1), #3
        maxpool_kernel_spatial = (1,1,1,1),
        maxpool_stride_spatial = (1,1,1,1),
        conv_bias=True,
        )
config=namedtuple('args',configDict)
config=config(**configDict)


for j in range(len(config.conv_dim)):
    print('Layer %d: Spat %d | Temp %d'%(j+1,*_get_layerOutputDim(j+1,config)))
 

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
        