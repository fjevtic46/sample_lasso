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

fp_data = '/mnt/research/compbio/krishnanlab/data/rnaseq/archs4/human_TPM/' # Private data, working on getting iris data.
fp = 'results/' # if you switched your directory to your scratch in the knitting script, change this to the scratch directory path
#sp = '/mnt/research/compbio/krishnanlab/projects/ImpApproaches/results/Lasso/rnaseq-rnaseq/'
sp = 'results/rnaseq-rnaseq/'
val   = np.arcsinh(np.load(fp_data + 'GPL570subset_ValExp.npy'))
test  = np.arcsinh(np.load(fp_data + 'GPL570subset_TstExp.npy'))
y_gene_idx = np.loadtxt(fp_data + 'GPL570subset_LINCS_ygenes_inds.txt',dtype=int)
y_val   = val[:,y_gene_idx]
y_test  = test[:,y_gene_idx]
trim = np.load(fp_data + 'trimmed_down_val_set_inds.npy')
y_val = y_val[trim,:]
parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type = float,
    help = 'alpha used for regression')
parser.add_argument('-date',type = str,
    help = 'Date of save path')
parser.add_argument('-model',type = str,
    help = 'gene or sample LASSO that was used. Use underscore to seperate words i.e. gene_LASSO or sample_LASSO')
args            = parser.parse_args()
alpha           = args.alpha
date            = args.date
model           = args.model

evals = ['val','test']
df = pd.DataFrame(columns=['metric','value', 'model', 'hyperparameter', 'split'])
test_missing  = np.load(fp + date +'/test_missing.npy').tolist()
for eval in evals:
	if eval == 'val':
		eval_data = y_val
		data_path = fp + date + '/y_pred_trimmed_%s_alpha_%f_LINCS_sample_lasso.npy' % (eval,alpha)
	elif eval == 'test':
		eval_data = y_test
		data_path = fp + date + '/y_pred_%s_alpha_%f_sample_lasso.npy' % (eval,alpha)
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
	df2['split'] = eval
	df2['data'] = 'trimmed'
	df = df.append(df2)
	print(eval)
df.to_csv(sp + '%s/trimmed_%s_gene_evaluations_%s_%f.csv' % (model,"test_and_val",model,alpha))

