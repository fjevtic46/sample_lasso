######################################################################################################################################################################
# This script is designed to take all of the output prediction files created from LASSO2 and knit them back together into an array with the same shape as y_val
######################################################################################################################################################################

from __future__ import division
import numpy as np
import time
import pandas as pd
import glob as glob
import os
import argparse

######################################################################################################################################################################

# Hard code some paths
fp_data           = '/mnt/research/compbio/krishnanlab/data/rnaseq/archs4/human_TPM/' # Private data, working to establish a iris dataset so others can work on it.
fp                = 'results/' # I recommend replacing this path with one in a scratch directory

######################################################################################################################################################################
data = "worm" # worm GPL570


if data == "GPL570":
    val   = np.arcsinh(np.load(fp_data + data + 'subset_ValExp.npy'))
    test  = np.arcsinh(np.load(fp_data + data + 'subset_TstExp.npy'))
elif data == "worm":
    val   = np.arcsinh(np.load("sample_data/" + data + '_ValExp.npy'))
    test  = np.arcsinh(np.load("sample_data/" + data + '_TstExp.npy'))

# Load Gene indices for genes in  both GPL and Archs4 data for X and splits based on GPL splits between platforms

if data == "GPL570":
    y_gene_idx = np.loadtxt(fp_data + data + 'subset_LINCS_ygenes_inds.txt',dtype=int)
    trim = np.load(fp_data + 'trimmed_down_val_set_inds.npy')
    
elif data == "worm":
    y_gene_idx = np.arange(50,100)

y_val   = np.transpose(val[:,y_gene_idx])
if data == "GPL570":
    y_val = y_val[:,trim]
y_test  = np.transpose(test[:,y_gene_idx])
evals = ['val','test']
print(np.shape(y_val))
print(np.shape(y_test))
parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type = float,
    help = 'alpha used during LASSO regression')
parser.add_argument('-date',type = str,
    help = 'Date of save path')
parser.add_argument('-split',type = str,
    help = 'Date of save path')
args            = parser.parse_args()
alpha         = args.alpha
date            = args.date
split           = args.split
if split == 'val':
	evals = ['val']
elif split == 'test':
	evals = ['test']
elif split == 'both':
	evals = ['val','test']
for eval in evals:
    if eval == 'val':
        missing = [] 
        start_model = 0
        last_model  = np.shape(y_val)[1]
        y_pred_val = [] 
        print(last_model)
        for model in range(start_model,last_model,1): 
            if os.path.isfile(fp + date + '/'+data+'_y_pred_trimmed_LINCS_%s_model_%i_alpha_%f_sample_Lasso.npy'%(eval,model,alpha)): 
                file_ = np.load(fp + date + '/'+data+'_y_pred_trimmed_LINCS_%s_model_%i_alpha_%f_sample_Lasso.npy'%(eval,model,alpha))
                y_pred_val.append(file_) 
            else: 
                missing.append(model) 
                continue 
        print(np.shape(y_pred_val))
        np.save(fp + date + '/'+data+'_y_pred_trimmed_%s_alpha_%f_LINCS_sample_lasso.npy'%(eval,alpha),y_pred_val)
        np.save(fp + date + '/'+data+'%f_trimmed_%s_LINCS_missing.npy'%(alpha,eval),missing)
        print(np.shape(missing))
        print('trimmed_missing', np.shape(missing))
    elif eval == 'test':
        missing = [] 
        start_model = 0
        last_model  = np.shape(y_test)[1]
        y_pred_test = [] 
        for model in range(start_model,last_model,1): 
            if os.path.isfile(fp + date + '/'+data+'_y_pred_%s_model_%i_alpha_%f_sample_Lasso.npy'%(eval,model,alpha)): 
                file_ = np.load(fp + date + '/'+data+'_y_pred_%s_model_%i_alpha_%f_sample_Lasso.npy'%(eval,model,alpha)) 
                y_pred_test.append(file_) 
            else: 
                missing.append(model) 
                continue 
        np.save(fp + date + '/'+data+'_y_pred_%s_alpha_%f_sample_lasso.npy'%(eval,alpha),y_pred_test)
        np.save(fp + date + '/'+data+'_%s_missing.npy'%eval,missing)
    else:
        print('Error occurred')

