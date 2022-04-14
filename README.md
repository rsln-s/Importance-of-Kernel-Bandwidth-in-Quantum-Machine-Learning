# Reproducibility materials for "Importance of Kernel Bandwidth in Quantum Machine Learning"

Repo structure:

- `code` contains Python scripts used to generate the data
- `reproduce-figures` contains Jupyter notebooks that generate the figures in the paper
- `data` contains the datasets used

## Requirements

Easiest way to reproduce my set up is to install anaconda3 and run
```
conda env create -f environment.yml
```

## Obtaining raw data

Full raw data can be downloaded here (~23GB): https://anl.box.com/s/v9hwxr0t9xlwgy9jfwzb0a0lhtysiogz

To reproduce figures, you have to first download the data (`results.zip`) and unzip it into `data` folder. 
