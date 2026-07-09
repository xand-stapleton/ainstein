# AInstein: Numerical Metrics on Manifolds  
This repository contains code for learning Einstein metrics ($R_{ij} = \lambda g_{ij}$) on varying topologies; including: spheres $S^n$, and the $S^2 \times \mathcal{P}$ topology for black hole solutions.   
  
This PINN unsupervised learning is run via the file `run.py`, where manifold properties and training hyperparameters are set using the `hyperparameters/hps.yaml` file. To instead train a supervised model (to either the identity function or a prespecified analytically known metric) run the file `run_supervised.py`, which uses hyperparameters from the same yaml file. Each topology has its own version of these files with respective filename modification.  

We recommend setting up a new environment for running of this package, the process for this is described in `environment/README.md`.  

## Running from the command line  
To run from the command line, enter the local directory of this package, ensure the environment is activated, set the run hyperparameters in `hyperparameters/hps.yaml`, and run the following code:  
### If using Weights & Biases:
```
python3 run.py -c=hyperparameters/hps.yaml
```
### ...otherwise:
```
wandb disabled
python3 run.py -c=hyperparameters/hps.yaml
```
Add the flags `--supervised` to run the model under the supervised regime against the respective analtyic metric, and `--identity` to use the respective identity metric for this supervised training instead.  

## Functionality
The package functionality is split according to: the model in `network/`, the losses in `losses/`, the sampling in `sampling/`, the geometric functions in `geometry/`, and some additonal useful functions in `helper_functions/`. The models are saved into the `runs` folder (the local filepath to this must first be set in `hps.yaml`), whilst the `seed_models` folder contains the pre-trained supervised models used as initialisations for the published results; more models can be trained and moved to this folder for different architecures and experiments.

Jupyter notebooks, with naming conventions of `examine_output.ipynb` plus the respective topology, are available in `notebooks/` which provides the testing functionality, and allows interactive visualisation of the trained models. Ensure the local filepath to the trained models is set correctly and follow internal instructions to set up the testing.   
  
## BibTeX Citation  
``` 
@article{Hirst:2025seh,
    author = "Hirst, Edward and Gherardini, Tancredi Schettini and Stapleton, Alexander G.",
    title = "{AInstein: Numerical Einstein Metrics via Machine Learning}",
    eprint = "2502.13043",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "QMUL-PH-25-04",
    doi = "10.1088/3050-287X/ae1117",
    journal = "AI Sci.",
    volume = "1",
    number = "2",
    pages = "025001",
    year = "2025"
}
@article{SchettiniGherardini:2026bdb,
    author = "Hirst, Edward and Schettini Gherardini, Tancredi and Stapleton, Alexander George",
    title = "{Black Hole Black Boxes: Numerical Black Hole Metrics via AInstein Neural Networks}",
    eprint = "2607.05489",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "MPIM-Bonn-2026",
    month = "7",
    year = "2026"
}
```

