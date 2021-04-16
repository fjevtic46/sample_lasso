import numpy as np
import sys, os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import time
import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
import numba
from numba import jit,njit

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-start_model',default = 0, type = int,
    help = 'The index of the target gene array to start with')
parser.add_argument('-Nmodels',default = 3, type = int,
    help = 'The number of target genes to train')
parser.add_argument('-alpha', default = 0.005, type = float,
        help = 'Regularization coefficient')
parser.add_argument('-date', type = str,
        help = 'save date')
parser.add_argument('-split', type = str,
        help = 'whether to run val or test split of data')
args             = parser.parse_args()
alpha            = args.alpha
N_Models         = args.Nmodels
split            = args.split
date             = args.date
start_model	     = int(args.start_model) * N_Models
stop_model       = start_model + N_Models




############################################################################################################

# Hard code some of the paths

fp_data = '/mnt/research/compbio/krishnanlab/data/rnaseq/archs4/human_TPM/' #Private data, working on getting iris data
fp_save = 'results/'# Recommend switching this to scratch directory path.




############################################################################################################

#This section will split the data up in train, val, and test sets
#@jit
#https://numba.pydata.org/numba-doc/latest/user/performance-tips.html 
#jit
@njit(fastmath = True)
def arc_sin(train,val,test):
    return np.arcsinh(train),np.arcsinh(val),np.arcsinh(test)

data = "GPL570" # worm GPL570

if data == "GPL570":
    t0 = time.time()
    train = np.load(fp_data + data + 'subset_TrnExp.npy')
    val   = np.load(fp_data + data + 'subset_ValExp.npy')
    test  = np.load(fp_data + data + 'subset_TstExp.npy')
    train,val,test = arc_sin(train,val,test)
elif data == "worm":
    t0 = time.time()
    train = np.load("sample_data/" + data + '_TrnExp.npy')
    val   = np.load("sample_data/" + data + '_ValExp.npy')
    test  = np.load("sample_data/" + data + '_TstExp.npy')
    train,val,test = arc_sin(train,val,test)

# Load Gene indices for genes in  both GPL and Archs4 data for X and splits based on GPL splits between platforms

if data == "GPL570":
    X_gene_idx = np.loadtxt(fp_data + data + 'subset_LINCS_Xgenes_inds.txt',dtype=int)
    y_gene_idx = np.loadtxt(fp_data+ data + 'subset_LINCS_ygenes_inds.txt',dtype=int)
    trim = np.load(fp_data + 'trimmed_down_val_set_inds.npy')
    
elif data == "worm":
    X_gene_idx = np.arange(0,50)
    y_gene_idx = np.arange(50,100)

# Make 
#jit
@njit(fast_math = True)
def t(data,train,val,X_gene_idx,y_gene_idx):
    X_train = np.transpose(train[:,X_gene_idx])
    y_train = np.transpose(train[:,y_gene_idx])
    X_val   = np.transpose(val[:,X_gene_idx])
    y_val   = np.transpose(val[:,y_gene_idx])
    if data == "GPL570":
        X_val   = X_val[:,trim]
        y_val   = y_val[:,trim]
    X_test  = np.transpose(test[:,X_gene_idx])
    y_test  = np.transpose(test[:,y_gene_idx])
    return X_train,y_train,X_val,y_val,X_test,y_test
    
X_train,y_train,X_val,y_val,X_test,y_test = t(data,train,val,X_gene_idx,y_gene_idx)
# standarize the data
std_scale = StandardScaler().fit(X_train)
X_train   = std_scale.transform(X_train)
y_train   = std_scale.transform(y_train)


if(split == "both"):
    split1 = "val"
    split2 = "test"

if split1 == 'val':
    if stop_model > X_val.shape[1]:
    	N_Models = N_Models - (stop_model-X_val.shape[1])
    stop_model = start_model + N_Models
    samples = range(start_model,stop_model,1)
    for sample in samples:
        t1 = time.time()
        clf = Lasso(alpha=alpha,fit_intercept=True,normalize=False,precompute=False,copy_X=True,max_iter=1000,tol=0.001,
                    warm_start=False,positive=False,random_state=None,selection='random')
        clf.fit(X_train,X_val[:,sample])
        print(clf.n_iter_)
        betas = clf.coef_
        np.save(fp_save + date + '/'+data+'_betas_trimmed_LINCS_val_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),betas)
        y_pred_val = clf.predict(y_train)
        np.save(fp_save + date + '/'+data+'_y_pred_trimmed_LINCS_val_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),y_pred_val)
        t2 = time.time()
        print('Time for model %i is:'%sample + str(datetime.timedelta(seconds=(t2-t1))))
else:
    print('Error Occurred')
if split2 == 'test':
    if stop_model > X_test.shape[1]:
    	N_Models = N_Models - (stop_model-X_test.shape[1])
    stop_model = start_model + N_Models
    samples = range(start_model,stop_model,1)
    for sample in samples:
        t1 = time.time()
        clf = Lasso(alpha=alpha,fit_intercept=True,normalize=False,precompute=False,copy_X=True,max_iter=1000,tol=0.001,
                    warm_start=False,positive=False,random_state=None,selection='random')
        clf.fit(X_train,X_test[:,sample])
        print(clf.n_iter_)
        betas = clf.coef_
        np.save(fp_save + date + '/'+data+'_betas_test_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),betas)
        y_pred_test = clf.predict(y_train)
        np.save(fp_save + date + '/'+data+'_y_pred_test_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),y_pred_test)
        t2 = time.time()
        print('Time for model %i is:'%sample + str(datetime.timedelta(seconds=(t2-t1))))
else:
    print('Error Occurred')
t3 = time.time()
print('Total time for completion: ' + str(datetime.timedelta(seconds=(t3-t0))))
print('alpha:  ', alpha)
print('date: ' ,date)
print('start model', start_model)
print('stop model', stop_model)
print('split',split)

                    
