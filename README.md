### Fair off-policy learning from observational data

This repository contains the code for our paper "Fair Off-Policy Learning from Observational Data".

![Plot Intuition](media/model_architecture.png)

#### Project structure 
- *data* contains the data generation files/ real-world data preprocessing
- *experiments* contains experiment code
- *models* contains codes for policy models + representation learning
- *hyperparam* contains hyperparameters and code for hyperparameter tuning
- *results* contains stored results and code for plotting graphs


#### Requirements
The project is build with python 3.9.7 and uses the packages listed in the file `requirements.txt`. In particular the following packages need to be installed to reproduce our results:
1. [Pytorch 1.10.0, Pytorch lightning 1.5.1] - deep learning models
2. [Optuna 2.10.0] - hyperparameter tuning
4. Other: Pandas 1.3.4, numpy 1.21.5, scikit-learn 1.0.1


#### Reproducing the experiments
The scripts running the experiments are contained in the `/experiments` folder. There are two directories, one for each dataset (synthetic = `/exp_sim` and real-world = `/exp_real`). Most experiments can be configured by a `.yaml` configuration file. Here, parameters for data generation (e.g., sample size, covariate dimension) as well as the methods used may be adjusted. The following base methods are available (for details see Appendix E):

- `untrained`: untrained model,
- `oracle`: oracle unconstrained policy, 
- `oracle_af`: oracle action fair policy,
- `fpnet`: Our framework FairPol.

The `fpnet` methods has two sub-specifications:
- `action_fair`: is either `auf` (action unfair) or `af_conf` (action fairness using domain confusion)
- `value_fair`: is either `vuf` (value unfair), `vef` (envy-free), or `vmm` (max-min)


#### Reproducing hyperparameter tuning
The code for hyperparameter tuning is contained in the `/hyperparam` folder. The main script running the tuning is `main.py`. The subfolders contain the configuration files and optimal parameters for the different experiments (synthetic = `/expsim`, real-world data = `/exp_real`). The optimal parameters are stored as `.yaml` files in the `/nuisance` subfolder (for estimating nuisance parameters) or in the `/policy_nets` subfolder (for FairPol).
