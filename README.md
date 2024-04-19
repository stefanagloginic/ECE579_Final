# ECE579
ECE579 Project (Dog Pose Estimation using Pytorch)


# Install Conda
Follow the steps to [install conda](https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install)

[Conda Documentation](https://conda.io/projects/conda/en/latest/user-guide/index.html)

# Create Environment
This repo has a shared environment.yml that stores the dependencies required for the project

```
conda env create -f environment.yml
```

Activate the new environment
```
conda activate ece_579_project
```

Make sure to update the env with latest

```
conda env update
```

# Install a new package 

Make sure you are in the ece_579_project before installing a package

```
conda install scipy
```

Update package

```
conda update scipy
```

List the packages in the environment

```
conda list
```

After an install you should update the `environment.yml` and push it to github along with your PR 

Read [Conda Package Docs](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html) for more info 

# Updating Environment 
Everytime an install is done and a new package is being used, the `environment.yml` must also be updated with the new additions

Active the environment 

```
conda activate ece_579_project
```

Update the yaml file (this makes it compatible with windows)
```
conda env export --from-history > environment.yml
```

# Jupyter Notebook in VSCode
The conda environment already has juypter package installed so we can run Ipython notebooks (.ipynb)

## Install the Jupyter Extension
[Jupyter Extension Docs](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

It should be in the recommended section 

once installed make sure you have already created the `ece_579_project` conda environment

open `test.ipynb` in visual studio

At the top right click the `Select Kernel` button

Select `ece_579_project` and then you can run the code by clicking `Run All` at the top 

Check all the outputs are correct, notice that pytorch is working

It can be useful to test if a package or code is working correct using notebooks

Notebooks can also be useful for data visualizations and data explorations


## Data Notes
is_multiple_dogs is a label we can use to remove images with multiple dogs
from the evaluation set (why? because our model can only evaluate one dog at a time)

joints can have incorrect values specifically in the validation set, 
it would be good to normalize them to [0, 0 , 0]. Or the image is discarded completely from the set. 

We need to validate that our images paths are valid and figure out how to discard images that are not valid. 