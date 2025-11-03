import numpy as np
from numpy import ndarray
from typing import Callable
from utils.base import Optim, EPS

class GradientDescent(Optim):
    """
    Implementation of the Batch Gradient Descent (I-Order) Algorithm.
    This optimizer uses the gradient of the *entire dataset* (a "batch")
    to compute each step. The gradient is provided by the `grad_func_callback`.
    Args:
        alpha (float): The learning rate, controlling the step size.
        alpha_optim (Optim, optional): An optimizer for performing line search
                                       to find the best `alpha` at each step.
    """
    def __init__(self, alpha: float = 0.01, alpha_optim: Optim | None = None) -> None:
        super().__init__()
        self.alpha_init = alpha  
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        return

    def _reset(self) -> None:
        super()._reset()
        self.alpha = self.alpha_init
        return

    def _next(
        self, 
        x: ndarray, 
        func_callback: Callable[[ndarray], float], 
        grad_func_callback: Callable[[ndarray], ndarray]
    ) -> ndarray:
        
        if isinstance(self.alpha_optim, Optim):
            current_grad = grad_func_callback(x)
            one_dim_func = lambda alpha_vec: func_callback(x - alpha_vec[0] * current_grad)
            optimized_alpha_vec = self.alpha_optim.optimize(
                x=np.array([self.alpha]), # Starting alpha
                func_callback=one_dim_func,
                grad_func_callback=lambda a: np.array([0.0]), 
                hessian_func_callback=lambda a: np.array([[0.0]]), 
                is_plot=False
            )
            self.alpha = optimized_alpha_vec[0]
        gradient = grad_func_callback(x)
        return x - self.alpha * gradient

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray] | None = None,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]
        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback)
            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x

class StochasticGradientDescent(Optim):
    """
    Implementation of Stochastic Gradient Descent (SGD).
    This optimizer updates weights using *one data point at a time*.
    The `optimize` loop is based on epochs, not gradient norm,
    as the stochastic gradient is noisy and may never reach zero.
    Args:
        alpha (float): The learning rate.
        n_epochs (int): The number of times to loop over the entire dataset.
    """
    def __init__(self, alpha: float = 0.01, n_epochs: int = 100) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.num_iter = n_epochs 

    def _reset(self) -> None:
        self.num_iter = 0
        return

    def _next(self, W: ndarray, x_i: ndarray, y_i: ndarray) -> ndarray:
        x_i = x_i.reshape(1, -1) 
        y_i = y_i.reshape(1, 1) 
        y_hat_i = x_i @ W
        E_i = y_hat_i - y_i
        stochastic_grad = x_i.T @ E_i
        return W - self.alpha * stochastic_grad

    def optimize(
        self,
        x: ndarray,
        X: ndarray,
        Y: ndarray,
        is_plot: bool = False,
        # These are ignored by this optimizer but needed for compatibility
        func_callback: Callable | None = None, 
        grad_func_callback: Callable | None = None,
        hessian_func_callback: Callable | None = None,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        """
        Runs the iterative SGD process for a fixed number of epochs.
        Args:
            x (ndarray): The initial weight vector `W`.
            X (ndarray): The *full* (augmented) feature matrix.
            Y (ndarray): The *full* target vector.
        """
        W = x.copy()
        plot_points: list[ndarray] = [W]
        N = X.shape[0]

        for epoch in range(self.n_epochs):
            # Shuffle the data for randomness
            permutation = np.random.permutation(N)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            
            for i in range(N):
                x_i = X_shuffled[i]
                y_i = Y_shuffled[i]
                W = self._next(W, x_i, y_i)
                
                if is_plot:
                    plot_points.append(W)
        
        self._reset()
        if is_plot:
            return W, plot_points
        return W

class MiniBatchGradientDescent(Optim):
    """
    Implementation of Mini-Batch Gradient Descent.
    Updates weights using a small batch of data at a time.
    This is a common and practical compromise between Batch GD and SGD.
    Args:
        alpha (float): The learning rate.
        n_epochs (int): The number of times to loop over the entire dataset.
        batch_size (int): The size of each mini-batch.
    """
    def __init__(self, alpha: float = 0.01, n_epochs: int = 100, batch_size: int = 32) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_iter = n_epochs

    def _reset(self) -> None:
        self.num_iter = 0
        return

    def _next(self, W: ndarray, x_batch: ndarray, y_batch: ndarray) -> ndarray:        
        m_batch = x_batch.shape[0]
        if m_batch == 0:
            return W
        y_hat_batch = x_batch @ W
        E_batch = y_hat_batch - y_batch
        mini_batch_grad = (1 / m_batch) * x_batch.T @ E_batch
        return W - self.alpha * mini_batch_grad

    def optimize(
        self,
        x: ndarray,
        X: ndarray,
        Y: ndarray,
        is_plot: bool = False,
        func_callback: Callable | None = None, 
        grad_func_callback: Callable | None = None,
        hessian_func_callback: Callable | None = None,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        """
        Runs the iterative mini-batch process for a fixed number of epochs.
        Args:
            x (ndarray): The initial weight vector `W`.
            X (ndarray): The *full* (augmented) feature matrix.
            Y (ndarray): The *full* target vector.
        """
        W = x.copy()
        plot_points: list[ndarray] = [W]
        N = X.shape[0]
        for epoch in range(self.n_epochs):
            permutation = np.random.permutation(N)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            for i in range(0, N, self.batch_size):
                x_batch = X_shuffled[i : i + self.batch_size]
                y_batch = Y_shuffled[i : i + self.batch_size]
                W = self._next(W, x_batch, y_batch)
                if is_plot:
                    plot_points.append(W)
        
        self._reset()
        if is_plot:
            return W, plot_points
        return W