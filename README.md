# sample_lasso

### Description

This module is based off of a computational biology project known as Expresto which explored the best methods to impute the gene expression 
profiles of incomplete microarray RNA samples. The full abstract can be read in the referenced link below. The method that was discovered to
perform best was lasso regression, where each microarray sample is a feature in a process known as Sample LASSO. My goal is to parallelize and
optimize aspects of the sample_lasso module in order to efficiently find the best hyperparameters for a set of samples.

Essentially, this module sample_lasso does 3 things:
1. Trains and fits a model to each validation and test sample, based on a set alpha hyperparameter
2. Performs prediction for each model, imputing the rest of the gene expression sequence
3. Uses a number of evaluation metrics to evaluate each model and gather all evaluations in an output file that is parsed to determine the best performing models.
4. This is repeated for every alpha being tested, with the goal of finding the best performing alpha.

### Data

This module uses data in a research space on the HPCC to contain the "true" data. I would have installed it and pushed it
however it is much too large to be included. However, I've included all files that are output from running 1 alpha. My plan
to demonstrate the benchmarking and optimization is to include the iris dataset for fast execution.

### Installation and execution

The module is run in 3 steps. It is 3 separate sbatch files that are run one after another, and only when the previous one is complete.
From the main sample_lasso directory, input commands into a dev-intel node on the HPCC as follows:

"sbatch LASSO2_v2_trimmed_LINCS.sb" <- will generate many jobs, wait for all jobs to complete before proceeding (Should take no more than 7 min)
"sbatch knitting_LASSO2.sb" <- knits all prediction arrays into big ones for both test and val sets.
"sbatch trimmed_gene_evaluations_LINCS_sample_lasso.sb" <- performs evaluation on test and val sets.

I also attached timing submission scripts for the 2 scripts I will focus on optimizing/parallelizing. Those will become relevant when
I begin benchmarking.

NOTE: If you wish to store all your slurm output and results in a scratch directory for temporary storage,
then replace all "results/" or "outputs/" paths in the three python files with the appropriate directory, and follow the structure
of those two folders when making the directories within them (this is only if you really want to output your results/slurm.outs in
an alternative location). 