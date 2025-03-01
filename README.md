# Deep-Double-Descent
Deep Double Descent is a phenomenon seen in deep learning models where the test error decreases, then increases, and finally decreases again as the model capacity or training time increases. This occurrence is especially noticeable in models with a high degree of overparameterization, challenging traditional notions of the bias-variance tradeoff.

The aim of this repository is to reproduce and expand on discoveries regarding Deep Double Descent through conducting various experiments on common datasets, serving as a guide for researchers exploring this phenomenon.

The repository is organized as follows:

```plaintext
deep-double-descent/  
├── LaTex/         # LaTeX document with the project report  
├── experiments/   # Experiment scripts and Jupyter notebooks for analysis  
├── results/       # Results obtained from experiments (processed on NGPU)  
├── models/        # Model definitions and architectures  
├── README.md      # README with the project description and overview  
