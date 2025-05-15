## Installing without a virtual env
To install the environment named ainstein, simply change to this directory in the shell and run
```
pip install -r environment/requirements.txt.
```


## Installing with a virtual env (conda or mamba -- recommended)
We STRONGLY recommend the use of a virtual environment. For example, with conda:
```
conda create -n ainstein python=3.12
```

Then, activate conda and install the required packages:
```
conda activate ainstein && pip install -r environment/requirements.txt
```
