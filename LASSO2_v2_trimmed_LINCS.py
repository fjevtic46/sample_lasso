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

############################################################################################################

# Hard code some of the paths

data_fp = '/mnt/research/compbio/krishnanlab/data/rnaseq/archs4/human_TPM/' #Private data, working on getting iris data
fp_save = 'results/'# Recommend switching this to scratch directory path.

############################################################################################################

#This section will split the data up in train, val, and test sets
t0 = time.time()
train = np.arcsinh(np.load(data_fp + 'GPL570subset_TrnExp.npy'))
val   = np.arcsinh(np.load(data_fp + 'GPL570subset_ValExp.npy'))
test  = np.arcsinh(np.load(data_fp + 'GPL570subset_TstExp.npy'))

# Load Gene indices for genes in  both GPL and Archs4 data for X and splits based on GPL splits between platforms
X_gene_idx = np.loadtxt(data_fp + 'GPL570subset_LINCS_Xgenes_inds.txt',dtype=int)
y_gene_idx = np.loadtxt(data_fp + 'GPL570subset_LINCS_ygenes_inds.txt',dtype=int)
trim = np.load(data_fp + 'trimmed_down_val_set_inds.npy')

# Make Splits
X_train = np.transpose(train[:,X_gene_idx])
y_train = np.transpose(train[:,y_gene_idx])
X_val   = np.transpose(val[:,X_gene_idx])
y_val   = np.transpose(val[:,y_gene_idx])
X_val   = X_val[:,trim]
y_val   = y_val[:,trim]
X_test  = np.transpose(test[:,X_gene_idx])
y_test  = np.transpose(test[:,y_gene_idx])


# standarize the data
std_scale = StandardScaler().fit(X_train)
X_train   = std_scale.transform(X_train)
y_train   = std_scale.transform(y_train)

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
        np.save(fp_save + date + '/betas_trimmed_LINCS_val_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),betas)
        y_pred_val = clf.predict(y_train)
        np.save(fp_save + date + '/y_pred_trimmed_LINCS_val_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),y_pred_val)
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
        np.save(fp_save + date + '/betas_test_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),betas)
        y_pred_test = clf.predict(y_train)
        np.save(fp_save + date + '/y_pred_test_model_%i_alpha_%f_sample_Lasso'%(sample,alpha),y_pred_test)
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

                
