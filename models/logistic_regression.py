# models/logistic_regression.py

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

class LogisticRegression(Algo):
    """
    Implementation of Logistic Regression for binary classification.
    
    Hypothesis: h(X) = sigmoid(XW)
    Loss Function: Binary Cross-Entropy
    """
    
    def __init__(self) -> None:
        self.W: ndarray | None = None  # Model weights (augmented)
        self.history: List[ndarray] = [] # To store convergence path

    def _add_bias(self, X: ndarray) -> ndarray:
        """Adds a bias (intercept) column of ones to the feature matrix X."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _sigmoid(self, z: ndarray) -> ndarray:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def __call__(self, X: ndarray) -> ndarray:
        """
        Makes predictions (hypothesis function).
        Returns the probability of class 1.
        """
        if self.W is None:
            raise ValueError("Model has not been trained yet. Call .train() first.")
        
        X_aug = self._add_bias(X)
        z = X_aug @ self.W
        return self._sigmoid(z)

    def predict(self, X: ndarray, threshold: float = 0.5) -> ndarray:
        """Predicts class labels (0 or 1)."""
        return (self(X) >= threshold).astype(int)

    def _loss(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> float:
        """
        [cite_start]Computes the Binary Cross-Entropy Loss[cite: 1256].
        Loss = -(1/N) * sum(y*log(h) + (1-y)*log(1-h))
        """
        N = X_aug.shape[0]
        z = X_aug @ W
        h = self._sigmoid(z)
        
        # Add epsilon for numerical stability (to avoid log(0))
        eps = 1e-9
        loss = -(1 / N) * np.sum(
            Y * np.log(h + eps) + (1 - Y) * np.log(1 - h + eps)
        )
        return loss

    def _gradient(self, W: ndarray, X_aug: ndarray, Y: ndarray) -> ndarray:
        """
        [cite_start]Computes the gradient of the Cross-Entropy Loss[cite: 1286].
        Gradient = (1/N) * X.T * (h - Y)
        """
        N = X_aug.shape[0]
        z = X_aug @ W
        h = self._sigmoid(z)
        E = h - Y # Error
        grad = (1 / N) * X_aug.T @ E
        return grad

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
        d = X_aug.shape[1]
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
        
        elif isinstance(optim, GradientDescent) or 'quasi_newton' in str(type(optim)):
            # These optimizers use the standard `optimize` signature
            loss_cb = lambda W: self._loss(
                W.reshape(-1, 1), X_aug, Y_train_col
            )
            grad_cb = lambda W: self._gradient(
                W.reshape(-1, 1), X_aug, Y_train_col
            ).flatten()
            
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
        """Calculates the final accuracy on the test set."""
        Y_pred = self.predict(X_test)
        Y_test_col = Y_test.reshape(-1, 1)
        accuracy = np.mean(Y_pred == Y_test_col)
        return accuracy