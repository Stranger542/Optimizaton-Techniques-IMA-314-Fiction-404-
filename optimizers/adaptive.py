import numpy as np
from numpy import ndarray
from typing import Callable
from utils.base import Optim, EPS

class Adagrad(Optim):
    """
    Implementation of the Adagrad (Adaptive Gradient) Algorithm.
    Adagrad adapts the learning rate for each parameter, performing
    larger updates for infrequent and smaller updates for frequent
    parameters. It does this by accumulating the *sum of squared gradients*
    for each parameter in an archive.
    Args:
        alpha (float): The initial (global) learning rate.
        epsilon (float): A small value for numerical stability,
                        added to the denominator.
    """

    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.epsilon = epsilon
        self.G: ndarray | None = None  
        self.alpha = alpha 
        return

    def _reset(self) -> None:
        super()._reset()
        self.G = None
        self.alpha = self.alpha_init
        return

    def _next(
        self, 
        x: ndarray, 
        gradient: ndarray
    ) -> ndarray:
        """Calculates the next position using the Adagrad update equations."""
        # Initialize G if it's None (first iteration)
        if self.G is None:
            self.G = np.zeros_like(x)
        
        self.G = self.G + gradient**2
        adaptive_lr = self.alpha / (np.sqrt(self.G) + self.epsilon)
        x_next = x - adaptive_lr * gradient
        return x_next

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray] | None = None,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        
        self.num_iter = 0 # Reset iteration count at the beginning
        plot_points: list[ndarray] = [x.copy()]
        
        # Initialize G
        if self.G is None:
            self.G = np.zeros_like(x)

        g = grad_func_callback(x) # Get initial gradient
        
        while np.linalg.norm(g) > EPS:
            self.num_iter += 1
            x = self._next(x, g) # Pass the gradient
            g = grad_func_callback(x) # Get new gradient for next check

            if is_plot:
                plot_points.append(x.copy())
            
            if self.num_iter > 50000:
                print(f"Warning: {self.__class__.__name__} reached max iterations.")
                break

        # Store iter count before resetting state
        final_iter_count = self.num_iter 
        self._reset()
        self.num_iter = final_iter_count # Restore it for printing

        if is_plot:
            return x, plot_points
        return x


class RMSProp(Optim):
    """
    Implementation of the RMSProp (Root Mean Square Propagation) Algorithm.
    RMSProp addresses Adagrad's aggressively diminishing learning rate
    by using an *Exponentially Weighted Moving Average (EWMA)*
    for the squared gradients, rather than a simple sum.
    Args:
        alpha (float): The learning rate.
        beta (float): The smoothing parameter for the EWMA (e.g., 0.9).
        epsilon (float): A small value for numerical stability.
    """

    def __init__(self, alpha: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.G: ndarray | None = None  

        self.alpha = alpha
        return

    def _reset(self) -> None:
        super()._reset()
        self.G = None
        self.alpha = self.alpha_init
        return

    def _next(
        self, 
        x: ndarray, 
        gradient: ndarray
    ) -> ndarray:
        """Calculates the next position using the RMSProp update equations."""
        # Initialize G if it's None (first iteration)
        if self.G is None:
            self.G = np.zeros_like(x)
        
        self.G = self.beta * self.G + (1 - self.beta) * (gradient**2)
        adaptive_lr = self.alpha / (np.sqrt(self.G) + self.epsilon)
        x_next = x - adaptive_lr * gradient
        return x_next

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray] | None = None,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        
        self.num_iter = 0 # Reset iteration count at the beginning
        plot_points: list[ndarray] = [x.copy()]
        
        # Initialize G
        if self.G is None:
            self.G = np.zeros_like(x)

        g = grad_func_callback(x) # Get initial gradient

        while np.linalg.norm(g) > EPS:
            self.num_iter += 1
            x = self._next(x, g) # Pass the gradient
            g = grad_func_callback(x) # Get new gradient for next check

            if is_plot:
                plot_points.append(x.copy())
            
            if self.num_iter > 50000:
                print(f"Warning: {self.__class__.__name__} reached max iterations.")
                break

        # Store iter count before resetting state
        final_iter_count = self.num_iter
        self._reset()
        self.num_iter = final_iter_count # Restore it for printing

        if is_plot:
            return x, plot_points
        return x

class Adam(Optim):
    """
    Implementation of the Adam (Adaptive Moment Estimation) Algorithm.
    Adam combines the ideas of Momentum (storing an EWMA of the 
    gradients, or 1st moment) and RMSProp (storing an EWMA of the 
    squared gradients, or 2nd moment).
    It also includes a *bias correction* step to account for the
    fact that the moment accumulators are initialized at zero.
    
    Args:
        alpha (float): The learning rate (e.g., 0.001).
        beta1 (float): The decay rate for the 1st moment (e.g., 0.9).
        beta2 (float): The decay rate for the 2nd moment (e.g., 0.99).
        epsilon (float): A small value for numerical stability.
    """

    def __init__(self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.M: ndarray | None = None
        self.G: ndarray | None = None  
        self.alpha = alpha
        return

    def _reset(self) -> None:
        super()._reset()
        self.M = None
        self.G = None
        self.alpha = self.alpha_init
        return

    def _next(
        self, 
        x: ndarray, 
        gradient: ndarray
    ) -> ndarray:
        """Calculates the next position using the Adam update equations."""
        # Initialize M and G if they're None (first iteration)
        if self.M is None:
            self.M = np.zeros_like(x)
        if self.G is None:
            self.G = np.zeros_like(x)
        
        k = self.num_iter # k is 1-indexed for bias correction
        
        self.M = self.beta1 * self.M + (1 - self.beta1) * gradient
        self.G = self.beta2 * self.G + (1 - self.beta2) * (gradient**2)
        
        # Bias correction
        M_hat = self.M / (1 - self.beta1**k)
        G_hat = self.G / (1 - self.beta2**k)
        
        x_next = x - (self.alpha / (np.sqrt(G_hat) + self.epsilon)) * M_hat
        
        return x_next

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray] | None = None,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        
        self.num_iter = 0 # Reset iteration count at the beginning
        plot_points: list[ndarray] = [x.copy()]
        
        # Initialize M and G
        if self.M is None:
            self.M = np.zeros_like(x)
        if self.G is None:
            self.G = np.zeros_like(x)

        g = grad_func_callback(x) # Get initial gradient

        while np.linalg.norm(g) > EPS:
            self.num_iter += 1 
            x = self._next(x, g) # Pass the gradient
            g = grad_func_callback(x) # Get new gradient for next check

            if is_plot:
                plot_points.append(x.copy())
            
            if self.num_iter > 50000:
                print(f"Warning: {self.__class__.__name__} reached max iterations.")
                break

        # Store iter count before resetting state
        final_iter_count = self.num_iter
        self._reset()
        self.num_iter = final_iter_count # Restore it for printing

        if is_plot:
            return x, plot_points
        return x