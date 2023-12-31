# SegrDNN
Codes for the paper entitled "Surface segregation machine-learned with inexpensive numerical fingerprint for the design of alloy catalysts" [*Mol. Catal.* **2023**, 541, 113096]. 

## Schematics

**Closed-loop hyper-parameter tuning of DNN model using Bayesian optimization (BO) and 10-fold CV:**

<p align="center">
	<img src="imgs/figure2a.png" alt="figure1" width="60%" height="60%"/>
</p>

The flowchart above was re-designed based on the figure in the original paper [*Mol. Catal.* **2023**, 541, 113096]. 

## Included Codes

####  Hyper-parameter tuning

- `using_keras_tuner.py`: for hyper-parameter tuning of DNN model for surface segregation energy (E<sub>segr</sub>) [link](https://github.com/dongjae-shin/SegrDNN/blob/main/codes/using_keras_tuner.py)
- `run.sh`: a shell script to prevent the python code from stopping [link](https://github.com/dongjae-shin/SegrDNN/blob/main/codes/run.sh)

#### Principal Component Analysis (PCA)

- `PCA_plots.ipynb` [link](https://github.com/dongjae-shin/SegrDNN/blob/main/codes/PCA_plots.ipynb)

#### Analyses based on predicted *E*<sub>segr</sub> values

- `Esegr_vs_CN_SHAP_screening.ipynb` [link](https://github.com/dongjae-shin/SegrDNN/blob/main/codes/Esegr_vs_CN_SHAP_screening.ipynb)

## Application of This Code

This code has been utilized in the following published paper:
1. D. Shin, G. Choi, C. Hong, and J. W. Han, *Mol. Catal.* 2023, 541, 113096 (https://doi.org/10.1016/j.mcat.2023.113096)
