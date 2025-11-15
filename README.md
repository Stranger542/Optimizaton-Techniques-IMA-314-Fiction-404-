# Optimization Algorithms from Scratch

A semester project implementation of **mathematical optimization algorithms**.  
This repository contains the **core Python code** for each algorithm along with **Jupyter Notebooks** demonstrating, visualizing, and comparing their performance.

---

## Authors

- **Sumit Hulke** (2023BCD0026)  
- **Suraj Sanjay Harlekar** (2023BCD0038)
- **Aryan Patil** (2023BCD0047)

---

## About This Project
 
We implement some optimization algorithms **from scratch** using only **NumPy**.

The main objectives are to:
- Understand the internal working of optimization techniques.
- Analyze their behavior on test functions
- Visualize their convergence trajectories.

The project separates:
- **Reusable algorithm logic**  in `.py` files  
- **Experimental analysis & visualization**  in `.ipynb` notebooks  

---

## Algorithms Implemented

### 1. First-Order Methods (Gradient-Based)
- **Gradient Descent (GD)** — Foundational batch optimization.  
- **Stochastic Gradient Descent (SGD)** — Processes one data point at a time.  
- **Mini-Batch SGD** — Balances speed and stability with small batches.

### 2. Momentum-Based Methods
- **Momentum GD** — Adds a velocity term to smooth updates.  
- **Nesterov Accelerated Gradient (NAG)** — A "look-ahead" version for faster convergence.

### 3. Adaptive Learning Rate Methods
- **Adagrad** — Adapts learning rate for each parameter.  
- **RMSProp** — Uses an exponentially weighted moving average of squared gradients.  
- **Adam** — Combines Momentum and RMSProp with bias correction.

### 4. Second-Order Methods
- **Newton’s Method** — Uses the Hessian for rapid convergence (one step for quadratics).  
- **Damped Newton’s Method** — Stabilizes updates when the Hessian isn’t positive definite.

### 5. Quasi-Newton Methods
- **BFGS (Broyden–Fletcher–Goldfarb–Shanno)** — Approximates the inverse Hessian efficiently.  
- **L-BFGS (Limited-memory BFGS)** — A memory-efficient version for high-dimensional problems using two-loop recursion.

### 6. Non-Differentiable & Other Methods
- **Sub-gradient Method** — Handles non-differentiable convex functions (e.g., Lasso regression).  
- **Line Search Techniques** — Used to determine optimal step size:
  - **Armijo Condition**
  - **Backtracking Line Search**

### 7. Regression_Models
- **Linear regression**  
  - **Baseline supervised learning model for continuous prediction.**
  - **Dataset Used: California Housing**
- **Ridge Regression (L2)**
  - **linear regression with L2 penalty to reduce overfitting.**
  - **Dataset Used: California Housing**
- **Lasso Regression (L1)**
  - **Adds an L1 penalty that increases sparsity**
  - **Dataset Used: California Housing**
- **Logistic Regression**
  - **Classification using sigmoid function.**
  - **Dataset Used: Breast Cancer Wisconsin Dataset**

---

## Project Structure

```bash
Optimization_Project/
├── optimizers/
│   ├── __init__.py
│   ├── gradient_descent.py         # GD, SGD, Mini-Batch
│   ├── line_search.py              # Armijo, Backtracking
│   ├── momentum.py                 # Momentum, Nesterov
│   ├── adaptive.py                 # Adagrad, RMSProp, Adam
│   ├── second_order.py             # Newton, Damped Newton
│   ├── quasi_newton.py             # BFGS, L-BFGS
│   └── non_differentiable.py       # Sub-gradient Method
│
├── notebooks/
│   ├── 01_Gradient_Descent       # GD, SGD
│   ├── 02_Momentum_Methods       # Momentum, NAG
│   ├── 03_Adaptive_Methods       # Adagrad, RMSProp, Adam
│   ├── 04_Second_Order_Methods   # Newton, Dampped Newton
│   ├── 05_Quasi_Newton_Methods   # BFGS, L-BFGS
│   └── 06_Subgradient_Method     # Lasso
│   └── 07_Regression_Models         # Linear, ridge , Lasso , Logistic
│
├── utils/
│   ├── __init__.py
│   ├── test_functions.py               # Test functions and  gradients
│   └── plot_helpers.py                 # Common plotting functions
│
├── requirements.txt                    
└── README.md                           
```

---

## Setup and Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Optimization_Project.git
cd Optimization_Project
```



### 2. Install Required Libraries
Install dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**
```
numpy
matplotlib
jupyter
scipy
```

## Example Visualization

Each notebook provides contour plots and convergence traces like:

- Gradient paths toward minima  
- Momentum trajectory smoothing  
- Adaptive methods vs. fixed learning rates  


