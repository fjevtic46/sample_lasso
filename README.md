# sample_lasso

### Description

This module is based off of a computational biology project known as Expresto which explored the best methods to impute the gene expression 
profiles of incomplete microarray RNA samples. The full abstract can be read in the referenced link below. The method that was discovered to perform best was lasso regression, where each microarray sample is a feature in a process known as Sample LASSO. My goal is to parallelize and optimize aspects of the sample_lasso module in order to efficiently find the best hyperparameters for a set of samples.

Essentially, this module sample_lasso does 3 things:
1. Trains a 