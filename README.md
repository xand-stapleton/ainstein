# AInstein: Numerical Metrics on Spheres  
This repository contains code for learning Einstein metrics ($R_{ij} = \lambda g_{ij}$) on Spheres of general dimension.  
  
The learning is run via the file `run.py`, where hyperparameters are set using the hyperparameters/hps.yaml file. To instead train a supervised model (to either the identity function or the round metric) run the file `run_supervised.py`, which uses hyperparameters from the same yaml file.  
  
### description of environment set up

### description of CL running

The package functionality is split according to the model in `network/model.py`, the losses in `losses/losses.py`, the sampling in `sampling/patch_sampling.py`, the geometric functions in `geometry/geometry.py`, and some additonal useful functions in `helper_functions/helper_functions.py`. The models are saved into the `runs` folder (the local filepath to this must first be set in `hps.yaml`), whilst the `runs_supervised` contains the pre-trained supervised models used as initialisations for the published results; more supervised models can be trained and moved to this folder for different architecures and experiments.

A jupyter notebook `examine_output.ipynb` is provided which provides the testing functionality, and allows interactive visualisation of the trained models. Ensure the local filepath is set correctly and follow internal instructions to set up the testing.  
  
# BibTeX Citation  
``` 
raise NotImplementedError()  
```

