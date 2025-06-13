<h1 style="font-size: 3em; text-align: center;">Theoretical and Empirical Analysis of Deep Double Descent</h1>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# Table of Contents

- [🎓 Introduction](#1-🎓-introduction)
- [📈 Main Results](#2-📈-main-results)
  - [🎯 Double Descent in Polynomial Regression](#️-double-descent-in-polynomial-regression)
  - [🧠 Deep Double Descent in Neural Networks](#️-deep-double-descent-in-neural-networks)
- [🧩 Key Findings](#3-📁-key-findings)
- [📁 Repository Structure](#4-📁-repository-structure)
- [🛠️ Installation](#5-🛠️-installation)
  - [✅ Requirements](#️-requirements)
  - [🔧 Setup Instructions](#️-setup-instructions)
- [📄 License](#6-📄-license)

---

# 1. 🎓 Introduction

Machine learning, particularly deep learning, has become essential across many fields. However, as models grow more complex, unexpected behaviors like Deep Double Descent emerge—challenging classical concepts such as the bias-variance trade-off and exposing gaps between theory and empirical results.

This project explores the computational and mathematical foundations behind Deep Double Descent, using both simple and advanced architectures. It aims to bridge traditional learning theory with modern trends, emphasizing interpretability, generalization, and the role of inductive biases in understanding overparameterized neural networks.

# 2. 📈 Main Results

Below are some of the key findings observed during our experiments. These results highlight the Deep Double Descent phenomenon in both simple polynomial regression and deep neural network settings.

### 🎯 Double Descent in Legendre Polynomial Regression

<div style="text-align: center;">
  <img src="img/legendre1DDD.png" alt="Polynomial Regression Double Descent" style="width: 95%;"/>
  <p><em><strong>Test error vs. Model complexity (number of params)</strong> showing the characteristic double descent curve in a simple regression using Legendre basis.</em></p>
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="img/legendre1.1.png" alt="Polynomial Regression Double Descent" style="width: 32%;"/>
  <img src="img/legendre1.2.png" alt="Another Polynomial Regression Image" style="width: 32%;"/>
  <img src="img/legendre1.3.png" alt="Third Polynomial Regression Image" style="width: 32%;"/>
</div>
<div style="text-align: center;">
  <em>Different approximations for Legendre basis across various parameterization zones, highlighting implicit regularization in the final overparameterized model (on the right), which results in a solution resembling the initial approximation.</em>
</div>



### 🧠 Deep Double Double Descent in Neural Networks

<div style="text-align: center;">
  <img src="img/model-epoch3CNNMNIST30k.png" alt="Deep Double Descent Heatmap" style="width: 95%;"/>
  <p><em><strong>Deep Double Descent by Model Capacity and Epochs</strong>, showing the test error (left image) in a heatmap versus model complexity (number of parameters) across varying epochs. The image highlights model-wise phenomena (horizontal lines) and epoch-wise phenomena (vertical lines). Additionally, the interpolation threshold — where the model achieves near-zero training error — can be observed in the train error heatmap (right image).</em></p>
</div>

<div style="text-align: center;">
  <img src="img/modelwisegh.png" alt="Epoch-wise Double Descent for PreActResNet" style="width: 95%;"/>
  <p><em><strong>Model-wise Double Descent for 3CNN</strong>, revealing how test error decreases as model capacity increases, with the peak corresponding to the interpolation threshold.</em></p>
</div>

<div style="text-align: center;">
  <img src="img/epoch-wisePreActResNet18(45,64).png" alt="Epoch-wise Double Descent for PreActResNet" style="width: 95%;"/>
  <p><em><strong>Epoch-wise Double Descent for PreActResNet</strong>, showing how test error decreases as the model approaches near-zero training error.</em></p>
</div>

<div style="text-align: center;">
  <img src="img/epochwisegh.png" alt="Epoch-wise Double Descent for PreActResNet" style="width: 95%;"/>
  <p><em><strong>Epoch-wise Double Descent for ResNet18</strong>, illustrating how test error can reach an even lower minimum after the second descent.</em></p>
</div>

***These results support the hypothesis that overparameterized models can generalize better in the overparameterized regime, challenging traditional assumptions of the bias-variance trade-off.***

---

## 3. 🧩 Key Findings

- **The classical bias-variance trade-off, as traditionally understood, does not hold in the overparameterized regime**.
- **Overparameterization ≠ Overfitting**.
- **The choice of basis functions in polynomial approximation is crucial for the quality of the final approximations**.
- **There are regions where paradoxically, increasing the number of parameters worsens model performance**.
- **Noise contributes to the emergence and intensity of the phenomena**.
- **Overparameterization can help achieve lower errors than those observed in the classical regime**.
- **The implicit biases of learning algorithms play a fundamental role by favoring simpler solutions, following the philosophy of Occam’s razor**.

---

## 4. 📁 Repository Structure

```plaintext
Deep-Double-Descent/   
├── experiments/                            # Experiment scripts and Jupyter notebooks for analysis and testing
│   ├── PlotNGPUresults                     # Jupyter notebook to plot results from NGPU experiments  
│   ├── Polynomial_approximation_(OLS_GD)   # Jupyter notebook for polynomial regression approximation (OLS + GD)
│   └── script                              # Shell script to launch neural network experiments
├── img/                                    # Images used in the README for documentation purposes
├── results/                                # Results obtained from experiments (processed on NGPU)  
├── src/                                    # Project source code for neural networks
│   ├── main/                               # Main fucntion to execute the training process 
│   ├── models/                             # Model architectures and definitions
│   └── utils/                              # Utility functions for data loading, noise addition, data splitting, training, and evaluation
├── requirements.txt                        # Python dependencies needed to run the project  
├── README.md                               # README with the project description and overview   
```

## 5. 🛠️ Installation

### ✅ Requirements

- Python 3.10+ (recommended)
- pip for dependency management

### 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/jantonioruiz/Deep-Double-Descent.git
cd Deep-Double-Descent

# (Optional) Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## 6. 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.