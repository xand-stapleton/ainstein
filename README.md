# AInstein: Numerical Metrics on Manifolds  
This repository contains code for learning Einstein metrics ($R_{ij} = \lambda g_{ij}$) on varying topologies; including: spheres $S^n$, the $S^2 \times \mathbb{R}^2$ topology of the Euclidean Schwarzschild solution, Lens spaces $L(3,n)$.   
  
The semi-supervised learning is run via the file `run.py`, where manifold properties and training hyperparameters are set using the `hyperparameters/hps.yaml` file. To instead train a supervised model (to either the identity function or a prespecified analytically known metric) run the file `run_supervised.py`, which uses hyperparameters from the same yaml file. Each topology has its own version of these files with respective filename modification.  

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

## Local Lorentzian Repeated-C Runs
For local single-patch Lorentzian experiments with Einstein-only training and
fixed Einstein constant `C`, use:

This local mode is a direct 2D patch run:
- no Schwarzschild 5D embedding pathway
- no Penrose-diagram sampling
- direct 2D ball-coordinate sampling and 2D Ricci-based Einstein loss

- Hyperparameters: `hyperparameters/hps_schwarzschild_local_lorentzian.yaml`
- Local launcher + evaluator: `run_local_eval.sh`

Typical flow:

```bash
bash run_local_eval.sh
```

Set `C` at the top of `run_local_eval.sh` before running.
The script will:
- run training 10 times with the same `C`
- rename each generated run directory to `runs/pos_0` ... `runs/pos_9` when `C > 0`,
  `runs/zero_0` ... `runs/zero_9` when `C = 0`, or
  `runs/neg_0` ... `runs/neg_9` when `C < 0`
- evaluate each run with `visualisation/report_schwarzschild_local_lambda.py`
- write per-run JSON metrics in each run folder as `test_einstein_loss.json`
- print aggregate summary metrics in the terminal

To evaluate one specific completed run directly:

```bash
python visualisation/report_schwarzschild_local_lambda.py --run-dir runs/<run_name>
```

The report script writes:
- per-run test Einstein metrics JSON in each run folder as
    `runs/<run_name>/test_einstein_loss.json`
    (includes `einstein_loss`, `det_g_mean`, and `det_g_std` over test points)

No CSV/Markdown aggregate files are written.

For `local_2d_mode` runs, the report script generates representative matplotlib
3D component plots on the 2D ball coordinates: one figure per pair for
`g_{ij}` and `R_{ij}`.

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
    month = "2",
    year = "2025"
}
```

