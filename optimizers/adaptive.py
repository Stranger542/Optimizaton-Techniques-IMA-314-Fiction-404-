# optimizers/adaptive.py

import numpy as np
from numpy import ndarray
from typing import Callable, list, tuple

# Import the base classes from your utils folder
from utils.base import Optim, EPS

# =============================================================================
# 1. Adagrad
# =============================================================================

class Adagrad(Optim):
    """
    Implementation of the Adagrad (Adaptive Gradient) Algorithm.

    Adagrad adapts the learning rate for each parameter, performing
    larger updates for infrequent and smaller updates for frequent
    parameters. It does this by accumulating the *sum of squared gradients*
    [cite_start]for each parameter in an archive `G`[cite: 2067, 2068].
    
    Args:
        [cite_start]alpha (float): The initial (global) learning rate[cite: 2054].
        epsilon (float): A small value for numerical stability,
                         [cite_start]added to the denominator[cite: 2056, 2069].
    """

    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.epsilon = epsilon
        [cite_start]self.G: ndarray | None = None  # Archive for squared gradients [cite: 2053]

        self.alpha = alpha # For compatibility, though it's mainly self.alpha_init
        return

    def _reset(self) -> None:
        """Resets the optimizer's state (iteration count and gradient archive)."""
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
        Calculates the next position using the Adagrad update equations.
        
        1. g(k) = grad(x(k))
        2. G(k) = G(k-1) + g(k)^2
        3. x(k+1) = x(k) - (alpha / (sqrt(G(k) + epsilon))) * g(k)
        """
        
        # [cite_start]1. Compute gradient [cite: 2065]
        gradient = grad_func_callback(x)
        
        # [cite_start]2. Accumulate squared gradient [cite: 2067]
        self.G = self.G + gradient**2
        
        # [cite_start]3. Update position [cite: 2069]
        # Calculate the adaptive learning rate for each parameter
        adaptive_lr = self.alpha / (np.sqrt(self.G) + self.epsilon)
        
        # Update position
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
        """
        Runs the iterative Adagrad process.
        """
        plot_points: list[ndarray] = [x]
        
        # [cite_start]Initialize the gradient accumulator G(0) to zeros [cite: 2053]
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


# =============================================================================
# 2. RMSProp
# =============================================================================

class RMSProp(Optim):
    """
    Implementation of the RMSProp (Root Mean Square Propagation) Algorithm.

    RMSProp addresses Adagrad's aggressively diminishing learning rate
    by using an *Exponentially Weighted Moving Average (EWMA)*
    [cite_start]for the squared gradients, rather than a simple sum[cite: 89, 90].
    
    Args:
        [cite_start]alpha (float): The learning rate[cite: 802].
        [cite_start]beta (float): The smoothing parameter for the EWMA (e.g., 0.9)[cite: 804].
        [cite_start]epsilon (float): A small value for numerical stability[cite: 803].
    """

    def __init__(self, alpha: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.beta = beta
        self.epsilon = epsilon
        # [cite_start]Archive for EWMA of squared gradients [cite: 801]
        self.G: ndarray | None = None  

        self.alpha = alpha
        return

    def _reset(self) -> None:
        """Resets the optimizer's state (iteration count and gradient archive)."""
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
        
        # [cite_start]1. Compute gradient [cite: 88]
        gradient = grad_func_callback(x)
        
        # [cite_start]2. Update EWMA of squared gradients [cite: 89]
        self.G = self.beta * self.G + (1 - self.beta) * (gradient**2)
        
        # [cite_start]3. Update position [cite: 91]
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
        """
        Runs the iterative RMSProp process.
        """
        plot_points: list[ndarray] = [x]
        
        # [cite_start]Initialize the gradient accumulator G(0) to zeros [cite: 801]
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


# =============================================================================
# 3. Adam
# =============================================================================

class Adam(Optim):
    """
    Implementation of the Adam (Adaptive Moment Estimation) Algorithm.

    Adam combines the ideas of Momentum (storing an EWMA of the 
    gradients, or 1st moment) and RMSProp (storing an EWMA of the 
    [cite_start]squared gradients, or 2nd moment)[cite: 816].
    
    It also includes a *bias correction* step to account for the
    [cite_start]fact that the moment accumulators are initialized at zero[cite: 826, 827].
    
    Args:
        [cite_start]alpha (float): The learning rate (e.g., 0.001)[cite: 816].
        [cite_start]beta1 (float): The decay rate for the 1st moment (e.g., 0.9)[cite: 818].
        [cite_start]beta2 (float): The decay rate for the 2nd moment (e.g., 0.99)[cite: 819].
        [cite_start]epsilon (float): A small value for numerical stability[cite: 817].
    """

    def __init__(self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # [cite_start]1st moment (Momentum) [cite: 816]
        self.M: ndarray | None = None
        # [cite_start]2nd moment (RMSProp) [cite: 816]
        self.G: ndarray | None = None  

        self.alpha = alpha
        return

    def _reset(self) -> None:
        """Resets the optimizer's state (iteration count and accumulators)."""
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
        
        # Get current iteration number (starts at 1)
        k = self.num_iter
        
        # [cite_start]2. Compute gradient [cite: 823]
        gradient = grad_func_callback(x)
        
        # [cite_start]3. Update 1st moment (Momentum) [cite: 824]
        self.M = self.beta1 * self.M + (1 - self.beta1) * gradient
        
        # [cite_start]4. Update 2nd moment (RMSProp) [cite: 825]
        self.G = self.beta2 * self.G + (1 - self.beta2) * (gradient**2)
        
        # [cite_start]5. Bias correction for 1st moment [cite: 826]
        M_hat = self.M / (1 - self.beta1**k)
        
        # [cite_start]6. Bias correction for 2nd moment [cite: 827]
        G_hat = self.G / (1 - self.beta2**k)
        
        # [cite_start]7. Update position [cite: 828]
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
        """
        Runs the iterative Adam process.
        """
        plot_points: list[ndarray] = [x]
        
        # [cite_start]Initialize the moment accumulators M(0) and G(0) to zeros [cite: 816]
        if self.M is None:
            self.M = np.zeros_like(x)
        if self.G is None:
            self.G = np.zeros_like(x)

        while np.linalg.norm(grad_func_callback(x)) > EPS:
            # Increment iteration counter *before* the step,
            # [cite_start]as Adam's bias correction starts at k=1 [cite: 821]
            self.num_iter += 1 
            x = self._next(x, grad_func_callback)

            if is_plot:
                plot_points.append(x)
            
            # Safety break for very long runs
            if self.num_iter > 50000:
                print("Adam reached max iterations (50,000).")
                break

        self._reset()
        if is_plot:
            return x, plot_points
        return x