# Optimization Algorithms from Scratch

A semester project implementation of various **mathematical optimization algorithms**.  
This repository contains the **core Python code** for each algorithm along with **Jupyter Notebooks** demonstrating, visualizing, and comparing their performance.

---

## Authors

- **Sumit Hulke** (2023BCD0026)  
- **Suraj Sanjay Harlekar** (2023BCD0038)
- **Aryan Patil** (2023BCD0047)

---

## About This Project

This project serves as an **educational exploration** into the field of **numerical optimization**.  
Based on the *Lecture 1–7* series, it implements foundational optimization algorithms **from scratch** using only **NumPy**.

The main objectives are to:
- Understand the internal working of optimization techniques.
- Analyze their behavior on test functions (e.g., simple quadratics, non-convex functions).
- Visualize their convergence trajectories.

The project separates:
- **Reusable algorithm logic** → in `.py` files  
- **Experimental analysis & visualization** → in `.ipynb` notebooks  

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
│   ├── 01_Gradient_Descent.ipynb       # Demos for GD, SGD
│   ├── 02_Momentum_Methods.ipynb       # Demos for Momentum, NAG
│   ├── 03_Adaptive_Methods.ipynb       # Demos for Adagrad, RMSProp, Adam
│   ├── 04_Second_Order_Methods.ipynb   # Demos for Newton
│   ├── 05_Quasi_Newton_Methods.ipynb   # Demos for BFGS, L-BFGS
│   └── 06_Subgradient_Method.ipynb     # Demo for Lasso
│   └── Regression_Models.ipynb 
│
├── utils/
│   ├── __init__.py
│   ├── test_functions.py               # Defines test functions and  gradients
│   └── plot_helpers.py                 # Common plotting utilities
│
├── requirements.txt                    # Dependencies
└── README.md                           # You are here
```

---

## Setup and Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Optimization_Project.git
cd Optimization_Project
```

### 2. Create a Virtual Environment
It’s recommended to use a virtual environment for dependency management.

**On macOS/Linux**
```bash
python3 -m venv venv
```

**On Windows**
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**On macOS/Linux**
```bash
source venv/bin/activate
```

**On Windows**
```bash
.env\Scriptsctivate
```

### 4. Install Required Libraries
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

> 💡 *Note:* `scipy` is included for benchmarking your implementations against `scipy.optimize`.

---

## Running the Experiments

All experiments and visualizations are available in **Jupyter Notebooks**.

### Step 1. Start the Jupyter Notebook Server
```bash
jupyter notebook
```
A new browser tab will open automatically.

### Step 2. Open the Notebooks
Navigate to the `notebooks/` folder in the Jupyter interface.

### Step 3. Run Experiments
Open any notebook (e.g., `01_Gradient_Descent.ipynb`) and execute cells using:
```
Shift + Enter
```

You’ll see:
- Test function setup  
- Optimizer import from the `optimizers/` module  
- 2D contour visualizations of convergence paths  
- Comparisons between algorithms (e.g., GD vs. Momentum)  

---

## Example Visualization

Each notebook provides contour plots and convergence traces like:

- Gradient paths toward minima  
- Comparison of learning rate effects  
- Momentum trajectory smoothing  
- Adaptive methods vs. fixed learning rates  

---

## License

This project is developed for educational purposes as part of an academic semester project.  
Feel free to use or adapt for learning and research purposes with proper credit.

---
