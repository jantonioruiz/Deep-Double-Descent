# Deep Double Descent
Deep Double Descent is a phenomenon seen in deep learning models where the test error decreases, then increases, and finally decreases again as the model capacity or training time increases. This occurrence is especially noticeable in models with a high degree of overparameterization, challenging traditional notions of the bias-variance tradeoff.

The aim of this repository is to reproduce and expand on discoveries regarding Deep Double Descent through conducting various experiments on common datasets, serving as a guide for researchers exploring this phenomenon.

---

## 📁 Repository Structure

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

## 🛠️ Installation

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