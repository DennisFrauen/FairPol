### Fair off-policy learning from observational data

## Project structure 
- *data* contains the data generation
- *doc* contains overleaf files
- *experiments* contains experiment code
  - *experiment.py*: main experiment code
  - *config_test*: experiment configuration
- *models* contains codes for policy models + representation learning
  - *fp_net.py*: main code file
  - *unfair_baselines.py*: "unfair policy net" and nuisance parameter estimation
- *hyperparam* contains hyperparameter and code for hyperparameter tuning
- *results* contains stored results and code for plotting graphs
