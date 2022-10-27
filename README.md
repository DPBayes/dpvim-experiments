This folder contains the code in experiments for the paper: "DPVIm: Differentially Private Variational Inference Improved"

The conda environment used for these experiments is described in environment.yml and can be installed by calling `conda env create -f environment.yml`

The `ukb` folder contains the code for the experiments involving the UKB data set (Sec. 4.2, Figure 1-3). NOTE, that we cannot share the UKB data can be shared. Therefore, the experiments cannot be rerun for this data set. The code is added here for transparency.

The `adult` folder contains the code for the experiments involving adult data set (Sec. 4.3, Figure 4).

The `linear_regression` folder contains the code for the experiments comparing full-rank vanilla DPVI to DPVI with aligned gradients (Sec 4.4, Figure 5).

Each subfolder also has a separate `README.md` file with further information.
