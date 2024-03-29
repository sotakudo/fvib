# Flexible Variational Information Bottleneck
The implementation codes of [Flexible Variational Information Bottleneck: Achieving Diverse Compression with a Single Training](https://arxiv.org/abs/2402.01238).

## To use in a project
See `demo.ipynb` for simple description of the usage.  
The file `utils.py` contains functions for the learning.  
The file `fvib.py` contains modules for FVIB and VIB.  
The file `loss.py` contains loss functions for FVIB, VIB and the Taylor approximaition of the VIB objective.  
The file `calibration.py` contains a module for continuous optimization of $\beta$ in FVIB and a module to calculate ECE.  

## Citation
If you find this library useful please consider citing our paper.
