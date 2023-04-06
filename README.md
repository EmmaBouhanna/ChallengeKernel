# ChallengeKernel
Challenge Kernel Methods

### Configuration
The file config.py contains several parameters that you can change, for example :
- N_FOLDS : number of folds used in crossvalidation
- WL_ITERS : the number of iterations performed in the Weisfeiler Lehman algorithm (default = 5)
- PRECOMPUTE_KERNELS : if set to True, will look for precomputed kernels K(training_data, training_data), and K(test_data, training_data)


### Reproducibility
To reproduce the experiment that scored best on the testing leaderboard, simply execute :
```
python start.py
