import numpy as np
from numpy import ndarray
from typing import Callable, List, Tuple
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
        grad_func_callback: Callable[[ndarray], ndarray]
    ) -> ndarray:
        """Calculates the next position using the Adagrad update equations.
        1. g(k) = grad(x(k))
        2. G(k) = G(k-1) + g(k)^2
        3. x(k+1) = x(k) - (alpha / (sqrt(G(k) + epsilon))) * g(k)
        """
    
        gradient = grad_func_callback(x)
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
        plot_points: list[ndarray] = [x]
        if self.G is None:
            self.G = np.zeros_like(x)

        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1
            x = self._next(x, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
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
        alpha (float): The learning rate[cite: 802].
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
        grad_func_callback: Callable[[ndarray], ndarray]
    ) -> ndarray:
        """
        Calculates the next position using the RMSProp update equations.
        1. g(k) = grad(x(k))
        2. G(k) = beta * G(k-1) + (1 - beta) * g(k)^2
        3. x(k+1) = x(k) - (alpha / (sqrt(G(k)) + epsilon)) * g(k)
        """
       
        gradient = grad_func_callback(x)
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
        plot_points: list[ndarray] = [x]
        if self.G is None:
            self.G = np.zeros_like(x)

        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1
            x = self._next(x, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
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
        grad_func_callback: Callable[[ndarray], ndarray]
    ) -> ndarray:
        """
        Calculates the next position using the Adam update equations.
        1. k = k + 1 (handled in `optimize` loop)
        2. g(k) = grad(x(k))
        3. M(k) = beta1 * M(k-1) + (1 - beta1) * g(k)
        4. G(k) = beta2 * G(k-1) + (1 - beta2) * g(k)^2
        5. M_hat = M(k) / (1 - beta1^k)
        6. G_hat = G(k) / (1 - beta2^k)
        7. x(k+1) = x(k) - (alpha / (sqrt(G_hat) + epsilon)) * M_hat
        """
        k = self.num_iter
        gradient = grad_func_callback(x)
        self.M = self.beta1 * self.M + (1 - self.beta1) * gradient
        self.G = self.beta2 * self.G + (1 - self.beta2) * (gradient**2)
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
        plot_points: list[ndarray] = [x]
        if self.M is None:
            self.M = np.zeros_like(x)
        if self.G is None:
            self.G = np.zeros_like(x)

        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1 
            x = self._next(x, grad_func_callback)

            if is_plot:
                plot_points.append(x)
            
            if self.num_iter > 50000:
                print("Adam reached max iterations (50,000).")
                break

        self._reset()
        if is_plot:
            return x, plot_points
        return x