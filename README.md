# AInstein: Numerical Metrics on Spheres  
This repository contains code for learning Einstein metrics ($R_{ij} = \lambda g_{ij}$) on spheres of general dimension, $S^n$.  
  
The semi-supervised learning is run via the file `run.py`, where manifold properties and training hyperparameters are set using the `hyperparameters/hps.yaml` file. To instead train a supervised model (to either the identity function or the round metric) run the file `run_supervised.py`, which uses hyperparameters from the same yaml file.  

We recommend setting up a new environemnt for running of this package, the process for this is described in `environment/README.md`.  

## Running from the command line  
To run from the command line, enter the local directory of this package, ensure the environment is activated, set the run hyperparameters in `hyperparameters/hps.yaml`, and run the following code:  
### If using Weights & Biases:
```
python3 run.py --hyperparameters=hyperparameters/hps.yaml
```
### ...otherwise:
```
wandb disabled
python3 run.py --hyperparameters=hyperparameters/hps.yaml
```

## Functionality
The package functionality is split according to: the model in `network/model.py`, the losses in `losses/losses.py`, the sampling in `sampling/patch_sampling.py`, the geometric functions in `geometry/geometry.py`, and some additonal useful functions in `helper_functions/helper_functions.py`. The models are saved into the `runs` folder (the local filepath to this must first be set in `hps.yaml`), whilst the `runs_supervised` folder contains the pre-trained supervised models used as initialisations for the published results; more supervised models can be trained and moved to this folder for different architecures and experiments.

A jupyter notebook `examine_output.ipynb` is provided which provides the testing functionality, and allows interactive visualisation of the trained models. Ensure the local filepath to the trained models is set correctly and follow internal instructions to set up the testing.  
  
## BibTeX Citation  
``` 
raise NotImplementedError()  
```

