# AInstein: Numerical Metrics on Spheres  
This repository contains code for learning Einstein metrics ($R_{ij} = \lambda g_{ij}$) on spheres of general dimension, $S^n$.  
  
The semi-supervised learning is run via the file `run.py`, where manifold properties and training hyperparameters are set using the `hyperparameters/hps.yaml` file. To instead train a supervised model (to either the identity function or the round metric) run the file `run_supervised.py`, which uses hyperparameters from the same yaml file.  

We recommend setting up a new environment for running of this package, the process for this is described in `environment/README.md`.  

## Running from the command line  
To run from the command line, enter the local directory of this package, ensure the environment is activated, set the run hyperparameters in `hyperparameters/hps.yaml`, and run the following code:  
### If using Weights & Biases:
```
python3 run.py --hyperparams=hyperparameters/hps.yaml
```
### ...otherwise:
```
wandb disabled
python3 run.py --hyperparams=hyperparameters/hps.yaml
```

## Functionality
The package functionality is split according to: the model in `network/`, the losses in `losses/`, the sampling in `sampling/`, the geometric functions in `geometry/`, and some additional useful functions in `helper_functions/helper_functions.py`. The models are saved into the `runs/` folder (the local filepath to this must first be set in `hps.yaml`), whilst the `runs_supervised/` folder contains the pre-trained supervised models used as initialisations for the published results; more supervised models can be trained and moved to this folder for different architecures and experiments.

A jupyter notebook `examine_output.ipynb` is provided which provides the testing functionality, and allows interactive visualisation of the trained models. Ensure the local filepath to the trained models is set correctly and follow internal instructions to set up the testing.   

## BibTeX Citation
> [!NOTE]
>
> Please cite this work if used by your project or otherwise redistributed! 
``` 
@article{Hirst_2025,
doi = {10.1088/3050-287X/ae1117},
url = {https://doi.org/10.1088/3050-287X/ae1117},
year = {2025},
month = {oct},
publisher = {IOP Publishing},
volume = {1},
number = {2},
pages = {025001},
author = {Hirst, Edward and Gherardini, Tancredi Schettini and Stapleton, Alexander G},
title = {AInstein: numerical Einstein metrics via machine learning},
journal = {AI for Science},
abstract = {A new semi-supervised machine learning package is introduced which successfully solves the Euclidean vacuum Einstein equations with a cosmological constant, without any symmetry assumptions. The model architecture contains subnetworks for each patch in the manifold-defining atlas. Each subnetwork predicts the components of a metric in its associated patch, with the relevant Einstein conditions of the form  being used as independent loss components (here , where n is the dimension of the Riemannian manifold, and the Einstein constant ). To ensure the consistency of the global structure of the manifold, another loss component is introduced across the patch subnetworks which enforces the coordinate transformation between the patches, , for an appropriate analytically known Jacobian J. We test our method for the case of spheres represented by a pair of patches in dimensions 2, 3, 4, and 5. In dimensions 2 and 3, the geometries have been fully classified. However, it is unknown whether a Ricci-flat metric can exist on spheres in dimensions 4 and 5. This work hints against the existence of such a metric.}
}

```

