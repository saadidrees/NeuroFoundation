#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:11:47 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""
"""
NOTE 1

Wav2Vec2's pre-training is known to be quite unstable.
It is advised to do a couple of test runs with a smaller dataset, 
i.e. --dataset_config_names clean clean, --dataset_split_names validation test
to find good hyper-parameters for learning_rate, batch_size, num_warmup_steps,
and the optimizer. A good metric to observe during training is the gradient norm 
which should ideally be between 0.5 and 2.
"""

# %% Import packages
import argparse
import math
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import os
import time
# import shutil
import numpy as np
import wandb



# import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
# from datasets import DatasetDict, concatenate_datasets, load_dataset
# from huggingface_hub import Repository, create_repo
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import h5py

import transformers
from transformers import (
    AdamW,
    SchedulerType,
    get_scheduler,
    is_wandb_available,
    set_seed,
)


import models.wav2vec2.feature_extraction_wav2vec2
from models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
import models.wav2vec2.modeling_wav2vec2
from models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTraining
import models.wav2vec2.configuration_wav2vec2
from models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

# from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.utils import get_full_repo_name, send_example_telemetry

wandb.init()
wandb.login(key=[2bdd9e765e05763a9846ddeab362ec20dfbc8138])


logger = get_logger(__name__)

dset_train = []
dset_val = []

CLUSTER = 0

if CLUSTER==0:
    fname_dset = '/home/saad/data/analyses/wav2vec2/datasets/dataset_train.h5'
else:
    fname_dset = 'home/sidrees/scratch/NeuroFoundation/data/datasets/dataset_train.h5'
    
with h5py.File(fname_dset,'r') as f:
    for i in range(len(f['dset_train'].keys())):
        rgb = {}
        idx= str(i)
        for key in f['dset_train'][idx].keys():
            if key=='input_values':
                rgb[key] = np.array(f['dset_train'][idx][key],dtype='float32')
            else:
                rgb[key] = np.array(f['dset_train'][idx][key])
        dset_train.append(rgb)
        
    for i in range(len(f['dset_val'].keys())):
        rgb = {}
        idx= str(i)
        for key in f['dset_val'][idx].keys():
            if key=='input_values':
                rgb[key] = np.array(f['dset_val'][idx][key],dtype='float32')
            else:
                rgb[key] = np.array(f['dset_val'][idx][key])
        dset_val.append(rgb)
context_len = dset_train[0]['input_values'].shape[0]

# %% Input arguments
from collections import namedtuple
argsDict = dict(
    dataset_name='Isma/librispeech_1000_seed_42', #"librispeech_asr",
    dataset_config_names=["clean","clean"],
    dataset_split_names=['train.clean.100'],
    model_name_or_path="patrickvonplaten/wav2vec2-base-v2",
    
    dim_inputSpat = 128,
    dim_inputTemp = context_len,
    
    feat_extract_norm_axis = 'None', # spatial, spatial-temporal, channels, None
   
    max_train_steps=None,
    num_train_epochs=1000,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=1,  # 8

    lr_scheduler_type='linear',     # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau']
    num_warmup_steps=32000,#32000, #32000,
    learning_rate=1e-3, #0.005,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-06,
    
    max_gumbel_temperature=2,
    min_gumbel_temperature=0.5,
    gumbel_temperature_decay=0.999995,

    logging_steps=1,
    saving_steps=1000,
    saving_epochs=25,
    gradient_checkpointing=True,
    mask_time_prob=0.65, #0.65,
    mask_time_length=3,        # Having this smaller (6) improves validation losses - but why?

    preprocessing_num_workers=None,
    validation_split_percentage=1,
    # audio_column_name='audio',
    config_name=None,
    train_cache_file_name=None,
    validation_cache_file_name=None,
    seed=0,
    pad_to_multiple_of=None,
    cache_dir=None,
    preprocessing_only=None,
    push_to_hub = False,
    )

if CLUSTER==0:
    # argsDict['output_dir'] = '/home/saad/data/analyses/wav2vec2/wav2vec2-2d-LN-'+argsDict['feat_extract_norm_axis']+'-LR-'+str(argsDict['learning_rate'])+'-contLen-'+str(context_len)+'-dropOut-0.3/'
    argsDict['output_dir'] = '/home/saad/data/analyses/wav2vec2/test/'
else:
    argsDict['output_dir'] = '/home/sidrees/scratch/NeuroFoundation/data/wav2vec2/wav2vec2-2d-LN-'+argsDict['feat_extract_norm_axis']+'-LR-'+str(argsDict['learning_rate'])+'-contLen-'+str(context_len)+'-dropOut-0.3/'

args=namedtuple('args',argsDict)
args=args(**argsDict)

os.makedirs(args.output_dir, exist_ok=False)

# %% init functions
@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 3

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        # mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch
    
    


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm



# %% Accelerator
# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
accelerator = Accelerator()
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    # datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # set up weights and biases if available
    if is_wandb_available():
        import wandb

        wandb.init(project=args.output_dir.split("/")[-1])
else:
    # datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
# if accelerator.is_main_process:
#     if args.push_to_hub and not args.preprocessing_only:
#         if args.hub_model_id is None:
#             repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
#         else:
#             repo_name = args.hub_model_id
#         create_repo(repo_name, exist_ok=True, token=args.hub_token)
#         repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)
#     elif args.output_dir is not None:
#         os.makedirs(args.output_dir, exist_ok=True)
# accelerator.wait_for_everyone()


# %% Model
# config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)
config = Wav2Vec2Config(dim_inputSpat=args.dim_inputSpat,dim_inputTemp=args.dim_inputTemp,feat_extract_norm_axis=args.feat_extract_norm_axis)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name_or_path)
feature_extractor.sampling_rate = 11
feature_extractor.do_normalize = False


# pretraining is only supported for "newer" stable layer norm architecture
# apply_spec_augment has to be True, mask_feature_prob has to be 0.0
if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
    raise ValueError(
        "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and"
        " ``config.feat_extract_norm='layer'"
    )

# initialize random model
model = Wav2Vec2ForPreTraining(config)
# model = Wav2Vec2ForPreTraining.from_pretrained(args.output_dir)
# model = model.cuda()


# Activate gradient checkpointing if needed
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# 4. Define data collator, optimizer and scheduler

mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length

data_collator = DataCollatorForWav2Vec2Pretraining(
    model=model,
    feature_extractor=feature_extractor,
    pad_to_multiple_of=args.pad_to_multiple_of,
    mask_time_prob=mask_time_prob,
    mask_time_length=mask_time_length,
)

"""
train_dataloader = DataLoader(
    vectorized_datasets["train"],
    shuffle=False, #True,
    collate_fn=data_collator,
    batch_size=args.per_device_train_batch_size,
)
eval_dataloader = DataLoader(
    vectorized_datasets["validation"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
)
"""
keys_to_remove = ('mouse_id','ophys_exp_id','stiminfo')
vec_dset_train = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_train))
vec_dset_val = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_val))

train_dataloader = DataLoader(
    vec_dset_train,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=args.per_device_train_batch_size,
)
eval_dataloader = DataLoader(
    vec_dset_val, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
)


# Optimizer
optimizer = AdamW(
    list(model.parameters()),
    lr=args.learning_rate,
    betas=[args.adam_beta1, args.adam_beta2],
    eps=args.adam_epsilon,
)

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Scheduler and math around the number of training steps.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)   # after how many batches should stuff be updated

if args.max_train_steps is None:
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
else:
    max_train_steps = args.max_train_steps
    

lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=max_train_steps,
)
# Afterwards we recalculate our number of training epochs
# args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# ---- Train
# if os.path.isdir(args.output_dir):
#     assert 1==2, 'Output directory already exists'
    # shutil.rmtree(args.output_dir)


total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(vec_dset_train)}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")
completed_steps = 0
starting_epoch = 0

# Only show the progress bar once on each machine.
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0
epoch=0
counter=0
t_epochs=[]
t_steps=[]

tloss_train = []
tloss_val = []
closs_train = []
closs_val = []
dloss_train = []
dloss_val = []
gnorm = []
lr_train = []
ppl_train = []

for epoch in range(starting_epoch, args.num_train_epochs):
    gc.collect()
    model.train()
    
    cp_file = args.output_dir+'cpt_'+str(0)
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            cp_file, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )

    # for step, batch in enumerate(train_dataloader):
    start_time_epoch = time.time()
    for step, batch in enumerate(train_dataloader):
        # print(step,batch['input_values'].shape[0])
        counter+=1
        start_time_step = time.time()
        
        # compute num of losses
        num_losses = batch["mask_time_indices"].sum()
        sub_attention_mask = batch.pop("sub_attention_mask", None)
        sub_attention_mask = (
            sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
        )
        percent_masked = num_losses / sub_attention_mask.sum()

        # forward
        torch.cuda.empty_cache()
        outputs = model(**batch)        # 
        
        """
        projected_quantized_states: [batch_size, seq_len, proj_codevector_dim]
        projected_states: [batch_size, seq_len, proj_codevector_dim]
        hidden_states: [num_attention_heads+1?][batch_size,seq_len,hidden_size]
        """
        

        # divide loss by gradient accumulation steps since gradients
        # are accumulated for multiple backward passes in PyTorch
        loss = outputs.loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        # make sure that `num_losses` is summed for distributed training
        # and average gradients over losses of all devices
        if accelerator.state.num_processes > 1:
            num_losses = accelerator.gather_for_metrics(num_losses).sum()
            gradient_multiplier = accelerator.state.num_processes / num_losses
            multiply_grads(model.module.parameters(), gradient_multiplier)
        else:
            multiply_grads(model.parameters(), 1 / num_losses)

        # update step
        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            # compute grad norm for monitoring
            scale = (
                accelerator.scaler._scale.item()
                if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                else 1
            )
            if accelerator.state.num_processes > 1:
                grad_norm = get_grad_norm(model.module.parameters(), scale)
            else:
                grad_norm = get_grad_norm(model.parameters(), scale)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()
            elif accelerator.is_local_main_process:
                progress_bar.write(
                    f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                )

            # update gumbel temperature
            gumbel_temperature = max(
                args.max_gumbel_temperature * args.gumbel_temperature_decay**completed_steps,
                args.min_gumbel_temperature,
            )
            if hasattr(model, "module"):
                model.module.set_gumbel_temperature(gumbel_temperature)
            else:
                model.set_gumbel_temperature(gumbel_temperature)

            progress_bar.update(1)
            completed_steps += 1

        # 6. Log all results
        if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
            loss.detach()
            outputs.contrastive_loss.detach()
            outputs.diversity_loss.detach()

            if accelerator.state.num_processes > 1:
                loss = accelerator.gather_for_metrics(loss).sum()
                outputs.contrastive_loss = accelerator.gather_for_metrics(outputs.contrastive_loss).sum()
                outputs.diversity_loss = accelerator.gather_for_metrics(outputs.diversity_loss).sum()
                percent_masked = accelerator.gather_for_metrics(percent_masked).sum()

            train_logs = {
                "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                "constrast_loss": outputs.contrastive_loss / num_losses,
                "div_loss": outputs.diversity_loss / num_losses,
                "%_mask_idx": percent_masked / accelerator.num_processes,
                "ppl": outputs.codevector_perplexity,
                "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                "temp": torch.tensor(gumbel_temperature),
                "grad_norm": torch.tensor(grad_norm),
            }
            
            tloss_train.append(train_logs['loss'].detach().cpu())
            closs_train.append(train_logs['constrast_loss'].detach().cpu())
            dloss_train.append(train_logs['div_loss'].detach().cpu())
            gnorm.append(train_logs['grad_norm'].detach().cpu())
            lr_train.append(train_logs['lr'].detach().cpu())
            ppl_train.append(train_logs['ppl'].detach().cpu())

            log_str = ""
            for k, v in train_logs.items():
                log_str += "| {}: {:.3e}".format(k, v.item())

            if accelerator.is_local_main_process:
                progress_bar.write(log_str)
                if is_wandb_available():
                    wandb.log(train_logs)
                    

        # save model every `args.saving_steps` steps
        if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
            if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )

            if (args.push_to_hub and epoch < args.num_train_epochs - 1) and accelerator.is_main_process:
                repo.push_to_hub(
                    commit_message=f"Training in progress step {completed_steps}",
                    blocking=False,
                    auto_lfs_prune=True,
                )

        # if completed steps > `args.max_train_steps` stop
        
        if completed_steps >= max_train_steps:
            torch.cuda.empty_cache()
            break
        
        torch.cuda.empty_cache()
        
        elapsed_time_step = time.time()-start_time_step
        t_steps.append(elapsed_time_step)
    
    # 7. Validate!
    model.eval()

    # init logs
    val_logs = {
        "val_loss": 0,
        "val_contrastive_loss": 0,
        "val_diversity_loss": 0,
        "val_num_losses": 0,
    }
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch.pop("sub_attention_mask", None)
            outputs = model(**batch)

        val_logs["val_loss"] += outputs.loss
        val_logs["val_contrastive_loss"] += outputs.contrastive_loss
        val_logs["val_diversity_loss"] += outputs.diversity_loss
        val_logs["val_num_losses"] += batch["mask_time_indices"].sum()


    # sum over devices in multi-processing
    if accelerator.num_processes > 1:
        val_logs = {k: accelerator.gather_for_metrics(v).sum() for k, v in val_logs.items()}

    val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

    tloss_val.append(val_logs['val_loss'].detach().cpu())
    closs_val.append(val_logs['val_contrastive_loss'].detach().cpu())
    dloss_val.append(val_logs['val_diversity_loss'].detach().cpu())

    log_str = ""
    for k, v in val_logs.items():
        log_str += "| {}: {:.3e}".format(k, v.item())

    if accelerator.is_local_main_process:
        progress_bar.write(log_str)
        if is_wandb_available():
            wandb.log(val_logs)

    # save model every `args.saving_epochs` epochs
    if (epoch+1) % args.saving_epochs == 0:
        cp_file = args.output_dir+'cpt_'+str(epoch+1)
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                cp_file, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
        fontsize=12
        fig,axs = plt.subplots(2,3,figsize=(20,10));axs=np.ravel(axs)
        axs[0].plot(np.array(tloss_train),label='train');axs[0].set_xlabel('Steps');axs[0].set_title('Total loss');axs[0].legend()
        axs[1].plot(np.array(closs_train),label='train');axs[1].set_xlabel('Steps');axs[1].set_title('contrastive loss');axs[0].legend()
        axs[2].plot(np.array(dloss_train),label='train');axs[2].set_xlabel('Steps');axs[2].set_title('diversity loss');axs[0].legend()
        axs[3].plot(np.array(lr_train));axs[3].set_xlabel('Steps');axs[3].set_title('lr')
        axs[4].plot(np.array(gnorm));axs[4].set_xlabel('Steps');axs[4].set_title('gradient norm')
        axs[5].plot(np.array(ppl_train));axs[5].set_xlabel('Steps');axs[5].set_title('Perplexity')
        fname_plot = os.path.join(args.output_dir,'fig_losses.png')
        fig.savefig(fname_plot)
        plt.show()
    
    elapsed_time_epoch = time.time()-start_time_epoch
    t_epochs.append(elapsed_time_epoch)

tloss_train = np.array(tloss_train)
closs_train = np.array(closs_train)
dloss_train = np.array(dloss_train)
lr_train = np.array(lr_train)
ppl_train = np.array(ppl_train)
gnorm = np.array(gnorm)
closs_val = np.array(closs_val)
dloss_val = np.array(dloss_val)
tloss_val = np.array(tloss_val)

# ---- Save training / val losses
import csv
fname_saveLosses = os.path.join(args.output_dir,'model_losses.csv')

header = ['train_tloss','train_closs','train_dloss','ppl_train','lr','gnorm','val_tloss','val_closs','val_dloss']
tloss_val_csv = np.empty(tloss_train.shape[0]);tloss_val_csv[:]=np.nan
tloss_val_csv[:tloss_val.shape[0]] = tloss_val
closs_val_csv = np.empty(closs_train.shape[0]);closs_val_csv[:]=np.nan
closs_val_csv[:closs_val.shape[0]] = closs_val
dloss_val_csv = np.empty(dloss_train.shape[0]);dloss_val_csv[:]=np.nan
dloss_val_csv[:dloss_val.shape[0]] = dloss_val

csv_data = np.concatenate((tloss_train[:,None],closs_train[:,None],dloss_train[:,None],ppl_train[:,None],lr_train[:,None],gnorm[:,None],tloss_val_csv[:,None],closs_val_csv[:,None],dloss_val_csv[:,None]),axis=1)
       
with open(fname_saveLosses,'w',newline='',encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(header) 
    for i in range(csv_data.shape[0]):
        csvwriter.writerow(csv_data[i]) 


"""
mdl_dir = '/home/saad/data/analyses/wav2vec2/wav2vec2-2d-inputNorm/'
fname_saveLosses = os.path.join(args.output_dir,'model_losses.csv')
tloss_train_csvread = []
closs_train_csvread = []
dloss_train_csvread = []
ppl_train_csvread = []
lr_train_csvread = []
gnorm_csvread = []
tloss_val_csvread = []
closs_val_csvread = []
dloss_val_csvread = []

with open(fname_saveLosses,'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        tloss_train_csvread.append(row['train_tloss'])
        closs_train_csvread.append(row['train_closs'])
        dloss_train_csvread.append(row['train_dloss'])
        ppl_train_csvread.append(row['ppl_train'])
        lr_train_csvread.append(row['lr'])
        gnorm_csvread.append(row['gnorm'])
        tloss_val_csvread.append(row['val_tloss'])
        closs_val_csvread.append(row['val_closs'])
        dloss_val_csvread.append(row['val_dloss'])

tloss_train = np.asarray(tloss_train_csvread,dtype='float64')
closs_train = np.asarray(closs_train_csvread,dtype='float64')
dloss_train = np.asarray(dloss_train_csvread,dtype='float64')
ppl_train = np.asarray(ppl_train_csvread,dtype='float64')
lr_train = np.asarray(lr_train_csvread,dtype='float64')
gnorm = np.asarray(gnorm_csvread,dtype='float64')
tloss_val = np.asarray(tloss_val_csvread,dtype='float64')
closs_val = np.asarray(closs_val_csvread,dtype='float64')
dloss_val = np.asarray(dloss_val_csvread,dtype='float64')
"""

# %% Plot losses
tloss_train = np.array(closs_train) + (config.diversity_loss_weight * np.array(dloss_train))
tloss_val = np.array(closs_val) + (config.diversity_loss_weight * np.array(dloss_val))

stepsPerEpoch = int(len(tloss_train)/args.num_train_epochs)

rgb = np.isnan(tloss_val)
tloss_val_plot = tloss_val[~rgb]
closs_val_plot = closs_val[~rgb]
dloss_val_plot = dloss_val[~rgb]

xaxis_val = np.floor(np.linspace(0,tloss_train.shape[0],num=tloss_val_plot.shape[0])).astype('int')

idx_toplot = np.arange(0,tloss_train.shape[0])  # tloss_train.shape[0]


fontsize=12
fig,axs = plt.subplots(2,3,figsize=(20,10));axs=np.ravel(axs)
axs[0].plot(tloss_train[idx_toplot],label='train');axs[0].plot(xaxis_val,tloss_val_plot,label='val');axs[0].set_xlabel('Steps');axs[0].set_title('Total loss');axs[0].legend()
axs[1].plot(closs_train[idx_toplot],label='train');axs[1].plot(xaxis_val,closs_val_plot,label='val');axs[1].set_xlabel('Steps');axs[1].set_title('contrastive loss');axs[0].legend()
axs[2].plot(dloss_train[idx_toplot],label='train');axs[2].plot(xaxis_val,dloss_val_plot,label='val');axs[2].set_xlabel('Steps');axs[2].set_title('diversity loss');axs[0].legend()
axs[3].plot(lr_train[idx_toplot]);axs[3].set_xlabel('Steps');axs[3].set_title('lr')
axs[4].plot(gnorm[idx_toplot]);axs[4].set_xlabel('Steps');axs[4].set_title('gradient norm')
axs[5].plot(ppl_train[idx_toplot]);axs[5].set_xlabel('Steps');axs[5].set_title('Perplexity')


# %% Old Stuff
"""
# %% Context vectors test: SVM
from sklearn import svm
import seaborn as sbn

cpts_all = np.arange(0,1050,50) #[0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500]

keys_to_remove = ('mouse_id','ophys_exp_id','stiminfo')


n_loops = 3

score1_stim_loop = np.zeros((len(cpts_all),n_loops));score1_stim_loop[:]=np.nan
score1_nonstim_loop = np.zeros((len(cpts_all),n_loops));score1_nonstim_loop[:]=np.nan
score1_total_loop = np.zeros((len(cpts_all),n_loops));score1_total_loop[:]=np.nan
score2_stim_loop = np.zeros((len(cpts_all),n_loops));score2_stim_loop[:]=np.nan
score2_nonstim_loop = np.zeros((len(cpts_all),n_loops));score2_nonstim_loop[:]=np.nan
score2_total_loop = np.zeros((len(cpts_all),n_loops));score2_total_loop[:]=np.nan
score_noiseAsStim_loop = np.zeros((len(cpts_all),n_loops));score_noiseAsStim_loop[:]=np.nan
score_noiseAsNonstim_loop = np.zeros((len(cpts_all),n_loops));score_noiseAsNonstim_loop[:]=np.nan


mdl_dir = '/home/saad/data/analyses/wav2vec2/wav2vec2-2d-inputNorm/'

c=0
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
        # config = model.config
    model = model.cuda()
    
    mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
    mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length

    
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
    
    vec_dset_noise = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_noisemovie))
    dataloader_noise = DataLoader(vec_dset_noise, collate_fn=data_collator, batch_size=32)
    
    
    hidden_size = config.hidden_size
    temp_dim = 32
    
    i=0
    for i in range(n_loops):
        print('Loop %d'%i)
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
        
        cv_noise_all = np.zeros((0,temp_dim,hidden_size))
        # model.eval()
        for step, batch in enumerate(dataloader_noise):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_noise_all = np.concatenate((cv_noise_all,rgb),axis=0)
    
        # for l in range(24):
        idx_entry = -1#l #-1#
        cv_svmtrain = cv_svmtrain_all[:,idx_entry,:]
        cv_svmtest1 = cv_svmtest1_all[:,idx_entry,:]
        cv_svmtest2 = cv_svmtest2_all[:,idx_entry,:]
        cv_noise = cv_noise_all[:,idx_entry,:]
        
        # cv_svmtrain = cv_svmtrain_all.reshape(-1,cv_svmtrain_all.shape[1]*cv_svmtrain_all.shape[2])
        # cv_svmtest1 = cv_svmtest1_all.reshape(-1,cv_svmtest1_all.shape[1]*cv_svmtest1_all.shape[2])
        # cv_svmtest2 = cv_svmtest2_all.reshape(-1,cv_svmtest2_all.shape[1]*cv_svmtest2_all.shape[2])
        # cv_noise = cv_noise_all.reshape(-1,cv_noise_all.shape[1]*cv_noise_all.shape[2])
        
        cv_svmtrain_norm = cv_svmtrain #(cv_svmtrain-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_svmtest1_norm = cv_svmtest1#(cv_svmtest1-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_svmtest2_norm = cv_svmtest2#(cv_svmtest2-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_noise_norm = cv_noise#(cv_noise-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        
        clf = svm.SVC()
        svm_fit = clf.fit(cv_svmtrain_norm, labels_svmtrain) # svm_fit = clf.fit(cv_svmtrain_norm[:nsamps_svmtrain], labels_svmtrain[:nsamps_svmtrain])
        print(svm_fit.score(cv_svmtrain_norm, labels_svmtrain)) # print(svm_fit.score(cv_svmtrain_norm[:nsamps_svmtrain], labels_svmtrain[:nsamps_svmtrain])) 
    
        labels_svmtest1_predict = svm_fit.predict(cv_svmtest1_norm) # labels_svmtest1_predict = svm_fit.predict(cv_svmtest1_norm[:nsamps_svmtest]); score=svm_fit.score(cv_svmtest1_norm[:nsamps_svmtest], labels_svmtest1[:nsamps_svmtest])
        score1_stim = (labels_svmtest1_predict[labels_svmtest1!=0] == labels_svmtest1[labels_svmtest1!=0]).sum()/len(labels_svmtest1[labels_svmtest1!=0])
        score1_nonstim = (labels_svmtest1_predict[labels_svmtest1==0] == labels_svmtest1[labels_svmtest1==0]).sum()/len(labels_svmtest1[labels_svmtest1==0])
        score1_total = (labels_svmtest1_predict == labels_svmtest1).sum()/len(labels_svmtest1)
        print('Test Set 1 Accuracy - Stim: %0.2f | NonStim: %0.2f | Total: %0.2f | idx_entry: %d' %(score1_stim,score1_nonstim,score1_total,idx_entry))
    
        labels_svmtest2_predict = svm_fit.predict(cv_svmtest2_norm)
        score2_stim = (labels_svmtest2_predict[labels_svmtest2!=0] == labels_svmtest2[labels_svmtest2!=0]).sum()/len(labels_svmtest2[labels_svmtest2!=0])
        score2_nonstim = (labels_svmtest2_predict[labels_svmtest2==0] == labels_svmtest2[labels_svmtest2==0]).sum()/len(labels_svmtest2[labels_svmtest2==0])
        score2_total = (labels_svmtest2_predict == labels_svmtest2).sum()/len(labels_svmtest2)
        print('Test Set 2 Accuracy - Stim: %0.2f | NonStim: %0.2f | Total: %0.2f' %(score2_stim,score2_nonstim,score2_total))
    
        # score1_stim = (labels_svmtest1_predict[labels_svmtest1==1] == labels_svmtest1[labels_svmtest1==1]).sum()/len(labels_svmtest1[labels_svmtest1==1])
        # score1_nonstim = (labels_svmtest1_predict[labels_svmtest1==2] == labels_svmtest1[labels_svmtest1==2]).sum()/len(labels_svmtest1[labels_svmtest1==2])
        # score1_total = (labels_svmtest1_predict == labels_svmtest1).sum()/len(labels_svmtest1)
        # print('Test Set 1 Accuracy - Stim 1: %0.2f | Stim 2: %0.2f | Total: %0.2f | idx_entry: %d' %(score1_stim,score1_nonstim,score1_total,idx_entry))
    
        labels_noise_predict = svm_fit.predict(cv_noise_norm)
        score2_noise_stim = (labels_noise_predict == 1).sum()/len(labels_noisemovie)
        score2_noise_nonstim = (labels_noise_predict == 0).sum()/len(labels_noisemovie)
        print('Noise - Classified as stim: %0.2f | Classified as nonstim %0.2f' %(score2_noise_stim,score2_noise_nonstim))
    
        score1_stim_loop[c,i] = score1_stim
        score1_nonstim_loop[c,i] = score1_nonstim
        score1_total_loop[c,i] = score1_total
        score2_stim_loop[c,i] = score2_stim
        score2_nonstim_loop[c,i] = score2_nonstim
        score2_total_loop[c,i] = score2_total
        score_noiseAsStim_loop[c,i] = score2_noise_stim
        score_noiseAsNonstim_loop[c,i] = score2_noise_nonstim
        
# %%

labels_scores = cpts_all
x_inter = 1
y_range = (0.0,0.9)
fig,axs=plt.subplots(1,3,figsize=(20,5));axs=np.ravel(axs)
axs[0].boxplot(score2_total_loop.T,labels=labels_scores);axs[0].set_ylim([0,1.1])
axs[0].set_yticks(np.arange(0,1.1,.1))
axs[0].set_ylim(y_range)
axs[0].set_xticks(np.arange(0,len(cpts_all),x_inter))
axs[0].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter,'int')])
axs[0].grid(axis='y',color=[.7,.7,.7],linestyle='--')
axs[0].set_title('total')

axs[1].boxplot(score2_stim_loop.T,labels=labels_scores);axs[1].set_ylim([0,1.1])
axs[1].set_yticks(np.arange(0,1.1,.1))
axs[1].set_ylim(y_range)
axs[1].grid(axis='y',color=[.7,.7,.7],linestyle='--')
axs[1].set_xticks(np.arange(0,len(cpts_all),x_inter))
axs[1].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter)])
axs[1].set_title('stim induced')


axs[2].boxplot(score2_nonstim_loop.T,labels=labels_scores);axs[2].set_ylim([0,1.1])
axs[2].set_yticks(np.arange(0,1.1,.1))
axs[2].set_ylim(y_range)
axs[2].grid(axis='y',color=[.7,.7,.7],linestyle='--')
axs[2].set_xticks(np.arange(0,len(cpts_all),x_inter))
axs[2].set_xticklabels(cpts_all[np.arange(0,len(cpts_all),x_inter)])
axs[2].set_title('non-stim induced')


# %% Context vectors test: KNN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import seaborn as sbn

cpts_all = np.arange(0,1000,50) #[0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500]

keys_to_remove = ('mouse_id','ophys_exp_id','stiminfo')


n_loops = 3

score1_stim_loop = np.zeros((len(cpts_all),n_loops));score1_stim_loop[:]=np.nan
score1_nonstim_loop = np.zeros((len(cpts_all),n_loops));score1_nonstim_loop[:]=np.nan
score1_total_loop = np.zeros((len(cpts_all),n_loops));score1_total_loop[:]=np.nan
score2_stim_loop = np.zeros((len(cpts_all),n_loops));score2_stim_loop[:]=np.nan
score2_nonstim_loop = np.zeros((len(cpts_all),n_loops));score2_nonstim_loop[:]=np.nan
score2_total_loop = np.zeros((len(cpts_all),n_loops));score2_total_loop[:]=np.nan
score_noiseAsStim_loop = np.zeros((len(cpts_all),n_loops));score_noiseAsStim_loop[:]=np.nan
score_noiseAsNonstim_loop = np.zeros((len(cpts_all),n_loops));score_noiseAsNonstim_loop[:]=np.nan

dim_temp = 24
dim_hidden = 768

mdl_dir = '/home/saad/data/analyses/wav2vec2/wav2vec2-2d-inputNorm/'

c=0
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
    model = model.cuda()
    
    mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
    mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length

    
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
    
    vec_dset_noise = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_noisemovie))
    dataloader_noise = DataLoader(vec_dset_noise, collate_fn=data_collator, batch_size=32)
    
    i=0
    for i in range(n_loops):
        print('Loop %d'%i)
        cv_svmtrain_all = np.zeros((0,dim_temp,dim_hidden))
        model.eval()
        for step, batch in enumerate(dataloader_svmtrain):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_svmtrain_all = np.concatenate((cv_svmtrain_all,rgb),axis=0)
                
                
        cv_svmtest1_all = np.zeros((0,dim_temp,dim_hidden))
        # model.eval()
        for step, batch in enumerate(dataloader_svmtest1):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_svmtest1_all = np.concatenate((cv_svmtest1_all,rgb),axis=0)
        
        
        cv_svmtest2_all = np.zeros((0,dim_temp,dim_hidden))
        # model.eval()
        for step, batch in enumerate(dataloader_svmtest2):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_svmtest2_all = np.concatenate((cv_svmtest2_all,rgb),axis=0)
        
        cv_noise_all = np.zeros((0,dim_temp,dim_hidden))
        # model.eval()
        for step, batch in enumerate(dataloader_noise):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch=batch.to(device='cuda')
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)
                rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
                cv_noise_all = np.concatenate((cv_noise_all,rgb),axis=0)
    
        # for l in range(24):
        idx_entry = -1#l #-1#
        cv_svmtrain = cv_svmtrain_all[:,idx_entry,:]
        cv_svmtest1 = cv_svmtest1_all[:,idx_entry,:]
        cv_svmtest2 = cv_svmtest2_all[:,idx_entry,:]
        cv_noise = cv_noise_all[:,idx_entry,:]
        
        # cv_svmtrain = cv_svmtrain_all.reshape(-1,cv_svmtrain_all.shape[1]*cv_svmtrain_all.shape[2])
        # cv_svmtest1 = cv_svmtest1_all.reshape(-1,cv_svmtest1_all.shape[1]*cv_svmtest1_all.shape[2])
        # cv_svmtest2 = cv_svmtest2_all.reshape(-1,cv_svmtest2_all.shape[1]*cv_svmtest2_all.shape[2])
        # cv_noise = cv_noise_all.reshape(-1,cv_noise_all.shape[1]*cv_noise_all.shape[2])
        
        cv_svmtrain_norm = cv_svmtrain #(cv_svmtrain-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_svmtest1_norm = cv_svmtest1#(cv_svmtest1-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_svmtest2_norm = cv_svmtest2#(cv_svmtest2-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        cv_noise_norm = cv_noise#(cv_noise-np.min(cv_svmtrain,axis=0)[None,:])/(np.max(cv_svmtrain,axis=0)[None,:]-np.min(cv_svmtrain,axis=0)[None,:])
        
        n_neighbors = len(np.unique(labels_svmtrain))
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(cv_svmtrain_norm, labels_svmtrain)

    
        labels_svmtest1_predict = neigh.predict(cv_svmtest1_norm) # labels_svmtest1_predict = svm_fit.predict(cv_svmtest1_norm[:nsamps_svmtest]); score=svm_fit.score(cv_svmtest1_norm[:nsamps_svmtest], labels_svmtest1[:nsamps_svmtest])
        score1_stim = (labels_svmtest1_predict[labels_svmtest1!=0] == labels_svmtest1[labels_svmtest1!=0]).sum()/len(labels_svmtest1[labels_svmtest1!=0])
        score1_nonstim = (labels_svmtest1_predict[labels_svmtest1==0] == labels_svmtest1[labels_svmtest1==0]).sum()/len(labels_svmtest1[labels_svmtest1==0])
        score1_total = (labels_svmtest1_predict == labels_svmtest1).sum()/len(labels_svmtest1)
        print('Test Set 1 Accuracy - Stim: %0.2f | NonStim: %0.2f | Total: %0.2f | idx_entry: %d' %(score1_stim,score1_nonstim,score1_total,idx_entry))
    
        labels_svmtest2_predict = neigh.predict(cv_svmtest2_norm)
        score2_stim = (labels_svmtest2_predict[labels_svmtest2!=0] == labels_svmtest2[labels_svmtest2!=0]).sum()/len(labels_svmtest2[labels_svmtest2!=0])
        score2_nonstim = (labels_svmtest2_predict[labels_svmtest2==0] == labels_svmtest2[labels_svmtest2==0]).sum()/len(labels_svmtest2[labels_svmtest2==0])
        score2_total = (labels_svmtest2_predict == labels_svmtest2).sum()/len(labels_svmtest2)
        print('Test Set 2 Accuracy - Stim: %0.2f | NonStim: %0.2f | Total: %0.2f' %(score2_stim,score2_nonstim,score2_total))
    
        # score1_stim = (labels_svmtest1_predict[labels_svmtest1==1] == labels_svmtest1[labels_svmtest1==1]).sum()/len(labels_svmtest1[labels_svmtest1==1])
        # score1_nonstim = (labels_svmtest1_predict[labels_svmtest1==2] == labels_svmtest1[labels_svmtest1==2]).sum()/len(labels_svmtest1[labels_svmtest1==2])
        # score1_total = (labels_svmtest1_predict == labels_svmtest1).sum()/len(labels_svmtest1)
        # print('Test Set 1 Accuracy - Stim 1: %0.2f | Stim 2: %0.2f | Total: %0.2f | idx_entry: %d' %(score1_stim,score1_nonstim,score1_total,idx_entry))
    
        labels_noise_predict = neigh.predict(cv_noise_norm)
        score2_noise_stim = (labels_noise_predict == 1).sum()/len(labels_noisemovie)
        score2_noise_nonstim = (labels_noise_predict == 0).sum()/len(labels_noisemovie)
        print('Noise - Classified as stim: %0.2f | Classified as nonstim %0.2f' %(score2_noise_stim,score2_noise_nonstim))
    
        score1_stim_loop[c,i] = score1_stim
        score1_nonstim_loop[c,i] = score1_nonstim
        score1_total_loop[c,i] = score1_total
        score2_stim_loop[c,i] = score2_stim
        score2_nonstim_loop[c,i] = score2_nonstim
        score2_total_loop[c,i] = score2_total
        score_noiseAsStim_loop[c,i] = score2_noise_stim
        score_noiseAsNonstim_loop[c,i] = score2_noise_nonstim
        
# %% Context vectors test: K-means
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import seaborn as sbn
from sklearn.decomposition import PCA


# model = Wav2Vec2ForPreTraining(config)
cpt = 700
mdl_dir = '/home/saad/data/analyses/wav2vec2/wav2vec2-2d-lr-smallernet/cpt_'+str(cpt)+'/'
model = Wav2Vec2ForPreTraining.from_pretrained(mdl_dir)
model = model.cuda()

mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length


data_collator = DataCollatorForWav2Vec2Pretraining(
    model=model,
    feature_extractor=feature_extractor,
    pad_to_multiple_of=args.pad_to_multiple_of,
    mask_time_prob=mask_time_prob,
    mask_time_length=mask_time_length,
)

keys_to_remove = ('mouse_id','ophys_exp_id','stiminfo')

vec_dset_svmtrain = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_svmtrain))
dataloader_svmtrain = DataLoader(vec_dset_svmtrain, collate_fn=data_collator, batch_size=32)

vec_dset_svmtest1 = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_svmtest1))
dataloader_svmtest1 = DataLoader(vec_dset_svmtest1, collate_fn=data_collator, batch_size=32)

vec_dset_svmtest2 = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_svmtest2))
dataloader_svmtest2 = DataLoader(vec_dset_svmtest2, collate_fn=data_collator, batch_size=32)

vec_dset_noise = list(map(lambda d: {k: v for k, v in d.items() if k not in keys_to_remove}, dset_noisemovie))
dataloader_noise = DataLoader(vec_dset_noise, collate_fn=data_collator, batch_size=32)

score1_stim_loop = []
score1_nonstim_loop = []
score1_total_loop = []
score2_stim_loop = []
score2_nonstim_loop = []
score2_total_loop = []
score_noiseAsStim_loop = []
score_noiseAsNonstim_loop = []

n_loops = 3
temp_dim = 32 #24
hidden_size = 480

for i in range(n_loops):
    print('Loop %d'%i)
    cv_svmtrain_all = np.zeros((0,temp_dim,hidden_size))
    model.eval()
    for step, batch in enumerate(dataloader_svmtrain):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch=batch.to(device='cuda')
            batch.pop("sub_attention_mask", None)
            outputs = model(**batch)
            rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
            # rgb = outputs['projected_states'].detach().cpu().numpy()
            cv_svmtrain_all = np.concatenate((cv_svmtrain_all,rgb),axis=0)
            
            
    cv_svmtest1_all = np.zeros((0,temp_dim,hidden_size))
    model.eval()
    for step, batch in enumerate(dataloader_svmtest1):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch=batch.to(device='cuda')
            batch.pop("sub_attention_mask", None)
            outputs = model(**batch)
            rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
            cv_svmtest1_all = np.concatenate((cv_svmtest1_all,rgb),axis=0)
    
    
    cv_svmtest2_all = np.zeros((0,temp_dim,hidden_size))
    model.eval()
    for step, batch in enumerate(dataloader_svmtest2):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch=batch.to(device='cuda')
            batch.pop("sub_attention_mask", None)
            outputs = model(**batch)
            rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
            cv_svmtest2_all = np.concatenate((cv_svmtest2_all,rgb),axis=0)
    
    cv_noise_all = np.zeros((0,temp_dim,hidden_size))
    model.eval()
    for step, batch in enumerate(dataloader_noise):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch=batch.to(device='cuda')
            batch.pop("sub_attention_mask", None)
            outputs = model(**batch)
            rgb = outputs['hidden_states'][-1].detach().cpu().numpy()
            cv_noise_all = np.concatenate((cv_noise_all,rgb),axis=0)

    
    for l in range(1):
        idx_entry = -1#l
        cv_svmtrain = cv_svmtrain_all[:,idx_entry,:]
        cv_svmtest1 = cv_svmtest1_all[:,idx_entry,:]
        cv_svmtest2 = cv_svmtest2_all[:,idx_entry,:]
        cv_noise = cv_noise_all[:,idx_entry,:]
        
        # cv_svmtrain = cv_svmtrain_all.reshape(-1,cv_svmtrain_all.shape[1]*cv_svmtrain_all.shape[2])
        # cv_svmtest1 = cv_svmtest1_all.reshape(-1,cv_svmtest1_all.shape[1]*cv_svmtest1_all.shape[2])
        # cv_svmtest2 = cv_svmtest2_all.reshape(-1,cv_svmtest2_all.shape[1]*cv_svmtest2_all.shape[2])
        # cv_noise = cv_noise_all.reshape(-1,cv_noise_all.shape[1]*cv_noise_all.shape[2])
        
        cv_svmtrain_norm = cv_svmtrain#(cv_svmtrain-np.mean(cv_svmtrain,axis=0)[None,:])/np.std(cv_svmtrain,axis=0)[None,:]
        cv_svmtest1_norm = cv_svmtest1#(cv_svmtest1-np.mean(cv_svmtrain,axis=0)[None,:])/np.std(cv_svmtrain,axis=0)[None,:]
        cv_svmtest2_norm = cv_svmtest2#(cv_svmtest2-np.mean(cv_svmtrain,axis=0)[None,:])/np.std(cv_svmtrain,axis=0)[None,:]
        cv_noise_norm = cv_noise#(cv_noise-np.mean(cv_svmtrain,axis=0)[None,:])/np.std(cv_svmtrain,axis=0)[None,:]
        
        n_neighbors = len(np.unique(labels_svmtrain))
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(cv_svmtrain_norm, labels_svmtrain)
        
        

        # fig,axs=plt.subplots(1,2,figsize=(15,5));fig.suptitle(str(idx_entry))
        # axs[0].hist([labels_svmtrain[labels_svmtrain==1],labels_svmtrain[labels_svmtrain==2],labels_svmtrain[labels_svmtrain==3],labels_svmtrain[labels_svmtrain==0]])
        # axs[1].hist([labels_svmtrain_predict[labels_svmtrain==1],labels_svmtrain_predict[labels_svmtrain==2],labels_svmtrain_predict[labels_svmtrain==3],labels_svmtrain_predict[labels_svmtrain==0]])
        # # axs[1].hist(labels_svmtrain)
        # plt.show()
        
       
    # fig,axs=plt.subplots(1,2,figsize=(15,5));fig.suptitle(str(idx_entry))
    # axs[0].hist(labels_svmtrain_predict[:nsamps_svmtrain])
    # axs[1].hist(labels_svmtrain[:nsamps_svmtrain])

    
    labels_svmtest1_predict = neigh.predict(cv_svmtest1_norm)
    score1_stim = (labels_svmtest1_predict[labels_svmtest1!=0] == labels_svmtest1[labels_svmtest1!=0]).sum()/len(labels_svmtest1[labels_svmtest1!=0])
    score1_nonstim = (labels_svmtest1_predict[labels_svmtest1==0] == labels_svmtest1[labels_svmtest1==0]).sum()/len(labels_svmtest1[labels_svmtest1==0])
    score1_total = (labels_svmtest1_predict == labels_svmtest1).sum()/len(labels_svmtest1)
    print('Test Set 1 Accuracy - Stim: %0.2f | NonStim: %0.2f | Total: %0.2f' %(score1_stim,score1_nonstim,score1_total))
    
    labels_svmtest2_predict = neigh.predict(cv_svmtest2_norm)
    score2_stim = (labels_svmtest2_predict[labels_svmtest2!=0] == labels_svmtest2[labels_svmtest2!=0]).sum()/len(labels_svmtest2[labels_svmtest2!=0])
    score2_nonstim = (labels_svmtest2_predict[labels_svmtest2==0] == labels_svmtest2[labels_svmtest2==0]).sum()/len(labels_svmtest2[labels_svmtest2==0])
    score2_total = (labels_svmtest2_predict == labels_svmtest2).sum()/len(labels_svmtest2)
    print('Test Set 2 Accuracy - Stim: %0.2f | NonStim: %0.2f | Total: %0.2f' %(score2_stim,score2_nonstim,score2_total))
    
    
    score1_stim_loop.append(score1_stim)
    score1_nonstim_loop.append(score1_nonstim)
    score1_total_loop.append(score1_total)
    score2_stim_loop.append(score2_stim)
    score2_nonstim_loop.append(score2_nonstim)
    score2_total_loop.append(score2_total)

    
    labels_noise_predict = neigh.predict(cv_noise_norm)
    score2_noise_stim = (labels_noise_predict == 1).sum()/len(labels_noisemovie)
    score2_noise_nonstim = (labels_noise_predict == 0).sum()/len(labels_noisemovie)
    print('Noise - Classified as stim: %0.2f | Classified as nonstim %0.2f' %(score2_noise_stim,score2_noise_nonstim))

    score_noiseAsStim_loop.append(score2_noise_stim)
    score_noiseAsNonstim_loop.append(score2_noise_nonstim)
    
    
    score1_stim_loop[c,i] = score1_stim
    score1_nonstim_loop[c,i] = score1_nonstim
    score1_total_loop[c,i] = score1_total
    score2_stim_loop[c,i] = score2_stim
    score2_nonstim_loop[c,i] = score2_nonstim
    score2_total_loop[c,i] = score2_total
    score_noiseAsStim_loop[c,i] = score2_noise_stim
    score_noiseAsNonstim_loop[c,i] = score2_noise_nonstim

    

fig,axs=plt.subplots(1,1,figsize=(10,5));axs=np.ravel(axs)
scores = [score1_stim_loop,score1_nonstim_loop,score1_total_loop,score2_stim_loop,score2_nonstim_loop,score2_total_loop,score_noiseAsStim_loop,score_noiseAsNonstim_loop]
labels_scores = ['stim 1','nonstim 1','total 1','stim 2','nonstim 2','total 2','noiseAsStim','noiseAsNontim']
axs[0].boxplot(scores,labels=labels_scores);axs[0].set_ylim([0,1.1])
axs[0].set_yticks(np.arange(0,1.1,.1))
axs[0].grid(axis='y',color=[.7,.7,.7],linestyle='--')
"""