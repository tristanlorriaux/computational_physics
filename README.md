# Computational physics project : Lennard-Jones gas/liquid transition modeling 

## Summary of the project 

Exploring the behavior of 2d Lennard-Jones systems under different conditions with the final objective to observe the phase transition via : 

* the implementation of the routines for calculating LJ energies and forces
* the calculation of the Virial pressure and compressibility
* the calculation of the particle mean-square displacements and diffusion constants
* the calculation of the local density
* routines for running individual and ensembles of MD simulations, sweeping parameter space and automatically including all available data sets into plots
![Capture d’écran 2022-03-01 141152](https://user-images.githubusercontent.com/57456860/156175157-37744dcb-bf51-4cd9-8545-ce558d4a0178.png)

## How to run our codes : 

**ALL OF THE USEFUL FUNCTIONS ARE IN *library1.py* or *library2.py* , IMPORT WITH :**

`from library1.py import *` or `from library2.py import *`

* `library1.py` is useful for `main_compressibility.py` and `main_density.py`
* `library2.py` is useful for `main_diffusion.py` and `main_istoherms.py`

Download all of the repository, execute one of the Jupyter `main_{}.py` of your choice. (Warning : you need `MPCMolecularDynamics.py` in the same repository as your Jupyter to correctly run the code !)

## How the project is constructed : 

Each part explore one of the aspect of the phase transition with more or less success : 

* Compressibility
* Diffusion and Mean Square Dispaclements
* Isotherms
* Local density

## How we worked on it : 

Two students have worked on it : Loic Malgrey and Tristan Lorriaux; Loic focused on MSD/Diffusion and Isotherms, while Tristan focused on Compressibility and Local Density. Both also worked on the other side of their tasks, to complete Jupyters with observations.
