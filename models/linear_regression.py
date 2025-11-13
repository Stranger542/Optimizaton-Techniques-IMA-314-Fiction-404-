# models/linear_regression.py

import numpy as np
from numpy import ndarray
from typing import Callable, List, Tuple

# Import the base class for algorithms
from utils.base import Algo, Optim

# Import the optimizers
from optimizers.gradient_descent import (
    GradientDescent, 
    StochasticGradientDescent, 
    MiniBatchGradientDescent
)
from optimizers.non_differentiable import SubGradientMethod

# =============================================================================
# 1. Base Class for Linear Regression Models
# =============================================================================

class LinearRegressionBase(Algo):
    """
    Base class for linear regression models.
    Inherits from the `Algo` base class in `utils/base.py`.
    """
    
    def __init__(self) -> None:
        self.W: ndarray | None = None  # Model weights (augmented)
        self.history: List[ndarray] = [] # To store convergence path

    def _add_bias(self, X: ndarray) -> ndarray:
        """Adds a bias (intercept) column of ones to the feature matrix X."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def __call__(self, X: ndarray) -> ndarray:
        """
        Makes predictions (hypothesis function).
        h(X) = X_aug @ W
        """
        if self.W is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        
        X_aug = self._add_bias(X)
        return X_aug @ self.W

    def _loss(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> float:
        """Abstract method for the loss function."""
        raise NotImplementedError
    
    def _gradient(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> ndarray:
        """Abstract method for the gradient (or sub-gradient)."""
        raise NotImplementedError

    def train(
        self, 
        X_train: ndarray, 
        Y_train: ndarray, 
        optim: Optim,
        is_plot: bool = False
    ) -> None:
        """
        Trains the model using the provided optimizer.
        """
        X_aug = self._add_bias(X_train)
        Y_train_col = Y_train.reshape(-1, 1) # Ensure Y is a column vector
        
        # Initialize weights
        d = X_aug.shape[1] # Number of features + 1 (for bias)
        self.W = np.random.randn(d, 1)
        
        # --- Dispatch based on optimizer type ---
        
        if isinstance(optim, (StochasticGradientDescent, MiniBatchGradientDescent)):
            # These optimizers have a custom `optimize` signature
            self.W, self.history = optim.optimize(
                x=self.W, 
                X=X_aug, 
                Y=Y_train_col, 
                is_plot=is_plot
            )
        
        elif isinstance(optim, (GradientDescent, SubGradientMethod)) or 'quasi_newton' in str(type(optim)):
            # These optimizers use the standard `optimize` signature
            
            # Create callbacks that pass data to the loss/gradient methods
            loss_cb = lambda W: self._loss(
                W.reshape(-1, 1), X_aug, Y_train_col
            )
            grad_cb = lambda W: self._gradient(
                W.reshape(-1, 1), X_aug, Y_train_col
            ).flatten() # Optimizers expect a 1D gradient
            
            W_flat, self.history = optim.optimize(
                x=self.W.flatten(),
                func_callback=loss_cb,
                grad_func_callback=grad_cb,
                hessian_func_callback=lambda W: np.identity(d), # Placeholder
                is_plot=is_plot
            )
            self.W = W_flat.reshape(-1, 1)
        else:
            raise TypeError(f"Optimizer type {type(optim).__name__} not supported by this model.")

    def test(self, X_test: ndarray, Y_test: ndarray) -> float:
        """Calculates the final loss (MSE) on the test set."""
        X_aug = self._add_bias(X_test)
        Y_test_col = Y_test.reshape(-1, 1)
        # Use the base MSE part of the loss, ignoring regularization
        N = X_aug.shape[0]
        E = X_aug @ self.W - Y_test_col
        mse = (1 / (2 * N)) * (E.T @ E)
        return mse.item()

# =============================================================================
# 2. Standard Linear Regression (MSE)
# =============================================================================

class LinearRegression(LinearRegressionBase):
    """
    Standard Linear Regression (Ordinary Least Squares).
    Loss = MSE
    """
    def _loss(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> float:
        N = X_aug.shape[0]
        E = X_aug @ W - Y
        mse = (1 / (2 * N)) * (E.T @ E)
        return mse.item()

    def _gradient(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> ndarray:
        N = X_aug.shape[0]
        E = X_aug @ W - Y
        grad = (1 / N) * X_aug.T @ E
        return grad

# =============================================================================
# 3. Ridge Regression (L2 Regularization)
# =============================================================================

class RidgeRegression(LinearRegressionBase):
    """
    Ridge Regression (Linear Regression + L2 Regularization).
    Loss = MSE + alpha * ||W||_2^2
    """
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha # Regularization strength (lambda)

    def _loss(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> float:
        N = X_aug.shape[0]
        E = X_aug @ W - Y
        mse = (1 / (2 * N)) * (E.T @ E)
        l2_penalty = (self.alpha / (2 * N)) * (W[1:].T @ W[1:]) # Don't penalize bias
        return (mse + l2_penalty).item()

    def _gradient(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> ndarray:
        N = X_aug.shape[0]
        E = X_aug @ W - Y
        mse_grad = (1 / N) * X_aug.T @ E
        
        # Don't penalize the bias term (W[0])
        l2_grad = (self.alpha / N) * W
        l2_grad[0] = 0 
        
        return mse_grad + l2_grad

# =============================================================================
# 4. Lasso Regression (L1 Regularization)
# =============================================================================

class LassoRegression(LinearRegressionBase):
    """
    Lasso Regression (Linear Regression + L1 Regularization).
    Loss = MSE + alpha * ||W||_1
    
    This loss function is NOT differentiable.
    It MUST be trained using a `SubGradientMethod`.
    """
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha # Regularization strength (lambda)

    def _loss(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> float:
        N = X_aug.shape[0]
        E = X_aug @ W - Y
        mse = (1 / (2 * N)) * (E.T @ E)
        l1_penalty = (self.alpha / N) * np.sum(np.abs(W[1:])) # Don't penalize bias
        return (mse + l1_penalty).item()

    def _gradient(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> ndarray:
        """
        This method computes the SUB-GRADIENT of the Lasso loss.
        """
        N = X_aug.shape[0]
        E = X_aug @ W - Y
        mse_grad = (1 / N) * X_aug.T @ E
        
        # Sub-gradient of the L1 norm: g(w) = sign(w)
        # (and any value in [-1, 1] if w_i = 0)
        l1_subgrad = (self.alpha / N) * np.sign(W)
        l1_subgrad[0] = 0 # Don't penalize bias
        
        return mse_grad + l1_subgrad
    

    def train(self, X_train: ndarray, Y_train: ndarray, optim: Optim, is_plot: bool = False) -> None:

        # --- Ensure correct optimizer ---
        from optimizers.non_differentiable import SubGradientMethod
        if not isinstance(optim, SubGradientMethod):
            raise TypeError("LassoRegression must be trained using SubGradientMethod.")

        # --- Preprocess ---
        X_aug = self._add_bias(X_train)
        Y_train_col = Y_train.reshape(-1, 1)

        # --- Init weights ---
        d = X_aug.shape[1]
        self.W = np.random.randn(d, 1)
        self.history = []   # store scalar losses ONLY

        # ---------------------------------------------------
        # Define callbacks for SubGradientMethod.optimize()
        # ---------------------------------------------------
        def loss_cb(W_flat):
            W = W_flat.reshape(-1, 1)
            E = X_aug @ W - Y_train_col

            mse = 0.5 * np.mean(E**2)
            l1 = (self.alpha / len(X_aug)) * np.sum(np.abs(W[1:]))

            loss = mse + l1
            self.history.append(float(loss))  # store scalar loss
            return float(loss)

        def grad_cb(W_flat):
            W = W_flat.reshape(-1, 1)
            E = X_aug @ W - Y_train_col

            mse_grad = (1 / len(X_aug)) * (X_aug.T @ E)
            l1_grad  = (self.alpha / len(X_aug)) * np.sign(W)
            l1_grad[0] = 0  # do not penalize bias

            g = mse_grad + l1_grad
            return g.flatten()

        # ---------------------------------------------------
        # Run optimizer
        # ---------------------------------------------------
        result = optim.optimize(
            x=self.W.flatten(),
            func_callback=loss_cb,
            grad_func_callback=grad_cb,
            hessian_func_callback=None,
            is_plot=is_plot
        )

        # Correctly unpack depending on is_plot flag
        if is_plot:
            W_flat, _ = result      # (x_best, path_points)
        else:
            W_flat = result         # x_best only

        self.W = W_flat.reshape(-1, 1)