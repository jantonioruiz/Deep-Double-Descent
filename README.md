<h1 style="font-size: 3em;">Deep Double Descent</h1>

# Table of Contents

- [🎓 Introduction](#1-🎓-introduction)
- [📈 Main Results](#2-📈-main-results)
  - [🎯 Double Descent in Polynomial Regression](#️-double-descent-in-polynomial-regression)
  - [🧠 Deep Double Descent in Neural Networks](#️-deep-double-descent-in-neural-networks)
- [📁 Repository Structure](#3-📁-repository-structure)
- [🛠️ Installation](#4-🛠️-installation)
  - [✅ Requirements](#️-requirements)
  - [🔧 Setup Instructions](#️-setup-instructions)
- [📄 License](#5-📄-license)

---

# 1. 🎓 Introduction

Machine learning, particularly deep learning, has become essential across many fields. However, as models grow more complex, unexpected behaviors like Deep Double Descent emerge—challenging classical concepts such as the bias-variance trade-off and exposing gaps between theory and empirical results.

This project explores the computational and mathematical foundations behind Deep Double Descent, using both simple and advanced architectures. It aims to bridge traditional learning theory with modern trends, emphasizing interpretability, generalization, and the role of inductive biases in understanding overparameterized neural networks.

# 2. 📈 Main Results

Below are some of the key findings observed during our experiments. These results highlight the Deep Double Descent phenomenon in both simple polynomial regression and deep neural network settings.

### 🎯 Double Descent in Legendre Polynomial Regression

![Polynomial Regression Double Descent](LaTex/img/experiments/Legendre1DDD.png)
Test error vs. model complexity showing the characteristic double descent curve in a simple regression using Legendre basis.

<div style="display: flex; justify-content: space-between;">
  <img src="LaTex/img/experiments/legendre1.1.png" alt="Polynomial Regression Double Descent" style="width: 32%;"/>
  <img src="LaTex/img/experiments/legendre1.2.png" alt="Another Polynomial Regression Image" style="width: 32%;"/>
  <img src="LaTex/img/experiments/legendre1.3.png" alt="Third Polynomial Regression Image" style="width: 32%;"/>
</div>
Different approximations across various parameterization zones, highlighting implicit regularization in the final overparameterized model (on the right), which results in a solution resembling the initial approximation.

### 🧠 Double Descent in Neural Networks

![Neural Network Double Descent](path/to/neural_dd_plot.png)

---

These results support the hypothesis that overparameterized models can generalize better under certain training regimes, challenging classical assumptions of the bias-variance trade-off.

## 3. 📁 Repository Structure

```plaintext
Deep-Double-Descent/  
├── LaTex/                                  # LaTeX document with the project report  
├── experiments/                            # Experiment scripts and Jupyter notebooks for analysis and testing
│   ├── PlotNGPUresults                     # Jupyter notebook to plot results from NGPU experiments  
│   ├── Polynomial_approximation_(OLS_GD)   # Jupyter notebook for polynomial regression approximation (OLS + GD)
│   └── script                              # Shell script to launch neural network experiments 
├── results/                                # Results obtained from experiments (processed on NGPU)  
├── src/                                    # Project source code for neural networks
│   ├── main/                               # Main fucntion to execute the training process 
│   ├── models/                             # Model architectures and definitions
│   └── utils/                              # Utility functions for data loading, noise addition, data splitting, training, and evaluation
├── requirements.txt                        # Python dependencies needed to run the project  
├── README.md                               # README with the project description and overview   
```

## 4. 🛠️ Installation

### ✅ Requirements

- Python 3.10 (recommended)
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

## 5. 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.