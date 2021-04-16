import numpy as np
import argparse
import os,sys
from sklearn.metrics import r2_score
from scipy import stats
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
import numba
from numba import jit

parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type = float,
    help = 'alpha used for regression')
parser.add_argument('-date',type = str,
    help = 'Date of save path')
parser.add_argument('-model',type = str,
    help = 'gene or sample LASSO that was used. Use underscore to seperate words i.e. gene_LASSO or sample_LASSO')
parser.add_argument('-Nmodels',type = int,
    help = 'To run val and test in parallel')

args            = parser.parse_args()
alpha           = args.alpha
date            = args.date
model           = args.model
n               = args.Nmodels


#######################################################################################################################
fp_data = '/mnt/research/compbio/krishnanlab/data/rnaseq/archs4/human_TPM/' # Private data.
fp = 'results/'
#sp = '/mnt/research/compbio/krishnanlab/projects/ImpApproaches/results/Lasso/rnaseq-rnaseq/'
sp = 'results/rnaseq-rnaseq/'

data = "worm" # worm GPL570

@jit
def arc_sin(val,test):
    return np.arcsinh(val),np.arcsinh(test)

if data == "GPL570":
    val   = np.load(fp_data + data + 'subset_ValExp.npy')
    test  = np.load(fp_data + data + 'subset_TstExp.npy')
    val,test = arc_sin(val,test)
elif data == "worm":
    val   = np.load("sample_data/" + data + '_ValExp.npy')
    test  = np.load("sample_data/" + data + '_TstExp.npy')
    val,test = arc_sin(val,test)
    
if data == "GPL570":
    y_gene_idx = np.loadtxt(fp_data + data + 'subset_LINCS_ygenes_inds.txt',dtype=int)
    trim = np.load(fp_data + 'trimmed_down_val_set_inds.npy')
    
elif data == "worm":
    y_gene_idx = np.arange(50,100)

y_val   = val[:,y_gene_idx]
y_test  = test[:,y_gene_idx]
if data == "GPL570":
    y_val = y_val[trim,:]
    

evals = ['val','test']
df = pd.DataFrame(columns=['metric','value', 'model', 'hyperparameter', 'split'])
test_missing  = np.load(fp + date +'/'+ data +'_test_missing.npy').tolist()
data_name = data

if n == 0:
    evaluation = "val"
    eval_data = y_val
    data_path = fp + date + '/'+data_name+'_y_pred_trimmed_%s_alpha_%f_LINCS_sample_lasso.npy' % (evaluation,alpha)
elif n == 1:
    evaluation = "test"
    eval_data = y_test
    data_path = fp + date + "/" + data_name + '_y_pred_%s_alpha_%f_sample_lasso.npy' % (evaluation,alpha)
    eval_data = np.delete(eval_data,test_missing,0)
else:
    print('Error Occurred')

data = np.load(data_path)
print(np.shape(eval_data),np.shape(data))
r2     = r2_score(eval_data,data,multioutput='raw_values')
mae    = mean_absolute_error(eval_data,data,multioutput='raw_values')
rmse   = np.sqrt(mean_squared_error(eval_data,data,multioutput='raw_values'))
cvrmse = rmse/np.mean(eval_data,axis=0)
rhos = [] # list to append rhos to in order to make spearmanr array
for idx in range(eval_data.shape[1]): # loops through each gene in y_val and calculates spearman correlation
	rho, p = stats.spearmanr(eval_data[:,idx],data[:,idx],axis=0)
	rhos.append(rho)
rho2s = []
for idx in range(eval_data.shape[1]): # loop through for each gene and calculate spearmanr and then make array of rhos
	rho2, p2 = stats.pearsonr(eval_data[:,idx],data[:,idx])
	rho2s.append(rho2)
cosines = []
for idx in range(eval_data.shape[1]): # list of cosine_similarity values for each gene
	cosine_ = -1 * cosine(eval_data[:,idx],data[:,idx]) + 1 # calculates cosine_similarity
	cosines.append(cosine_)
df2 = pd.DataFrame(np.vstack((r2,mae,rmse,cvrmse,rhos,rho2s,cosines)),index=['r2','mae','rmse','cvrmse','spearmanr','pearsonr','cosine'])
df2 = pd.melt(df2.T, value_vars = ['r2','mae','rmse','cvrmse','spearmanr','pearsonr','cosine'],var_name = 'metric',value_name = 'value')
df2['model'] = model
df2['hyperparameter'] = alpha
df2['split'] = evaluation
df2['data'] = 'trimmed'
df = df.append(df2)
print(evaluation)
df.to_csv(sp + '%s/'%(model) +data_name+"_" + evaluation +'_trimmed_%s_gene_evaluations_%s_%f.csv' % ("test_and_val",model,alpha))

