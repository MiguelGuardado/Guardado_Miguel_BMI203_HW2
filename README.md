# Project 2 - Clustering and Drug Discovery
## date completed : 02/19/2020

![BuildStatus](https://github.com/MiguelGuardado/Guardado_Miguel_BMI203_HW2/workflows/HW2/badge.svg?event=push)

In this assignment, I evaluate results from a high-throughput virtual screen against the SARS-CoV2 Spike protein / Human ACE2 interface.  There are two parts to this assignment / task that I completed. 
This repo will hold all the information needed to run clustering analysis on the molecular fingerprint data. This clustering preformed is a Partitioning(k-means) and Hierarchial clustering. This repo is desinged in 3 main
directories of the project, with extra files for writing up the results of my analysis.

* Part 1 - API and implementation
* Part 2 - Evaluating clustering

The data we are considering comes from [Smith and Smith, 2020](https://chemrxiv.org/articles/preprint/Repurposing_Therapeutics_for_the_Wuhan_Coronavirus_nCov-2019_Supercomputer-Based_Docking_to_the_Viral_S_Protein_and_Human_ACE2_Interface/11871402). In this study, they generated 6 Spike-Ace2 interface poses using MD simulations. They then docked ~10k small molecules against each protein conformation. Provided for you is the top (#1) pose for each ligand docked against one Spike-ACE2 interface conformation, as well as the corresponding SMILES string, AutoDock Vina score, and the “On” bits in the Extended Connectivity Fingerprint for that compound. These can all be found in ligand\_information.csv.


### docs
This section hold the API documentation for the cluster module and contains all the objects and functions calls in algs.py. Please see section below to access APU docs.

```
open docs/build/html/index.html
```
from the root directory of this project.

### clusters
This module is used to hold my algs sub-module for the clustering algorithim, this module holds a ligand class that reads the molecular fingerprints with ease. The clusterin algorthims can be
used for any type of data, as long as you have M observation (rows) by N feature(col) matrix. See API doc for more information about this. 


### test
this will be used to store all the unit tests for my algorithims, to confirm they work 

Testing is as simple as running
```
python -m pytest test/*
```
from the root directory of this project.

