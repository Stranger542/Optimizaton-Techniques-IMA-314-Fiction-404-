# optimizers/momentum.py

import numpy as np
from numpy import ndarray
from typing import Callable, list, tuple

# Import the base classes from your utils folder
from utils.base import Optim, EPS

# =============================================================================
# 1. Momentum Gradient Descent
# =============================================================================

class MomentumGradientDescent(Optim):
    """
    Implementation of Gradient Descent with Momentum (MGD).

    This optimizer accelerates GD by adding a "velocity" term (`v`)
    that accumulates past gradients, helping to smooth out oscillations
    [cite_start]and build speed in the correct direction [cite: 2605-2609].

    Args:
        alpha (float): The learning rate (e.g., 0.01).
        gamma (float): The momentum coefficient (e.g., 0.9), which
                       [cite_start]controls the decay of the velocity[cite: 2616].
    """

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.v: ndarray | None = None  # Velocity vector

        # Store initial values for reset
        self.alpha_init = alpha
        self.gamma_init = gamma
        return

    def _reset(self) -> None:
        """Resets the optimizer's state (iteration count and velocity)."""
        super()._reset()
        self.v = None
        self.alpha = self.alpha_init
        self.gamma = self.gamma_init
        return

    def _next(
        self, 
        x: ndarray, 
        grad_func_callback: Callable[[ndarray], ndarray]
    ) -> ndarray:
        """
        Calculates the next position using the Momentum GD update equations.
        
        1. [cite_start]v(k) = gamma * v(k-1) + alpha * grad(x(k)) [cite: 2613, 2643]
        2. [cite_start]x(k+1) = x(k) - v(k) [cite: 2614, 2644]
        """
        # 1. Compute gradient
        gradient = grad_func_callback(x)
        
        # 2. Update velocity
        # We assume self.v has been initialized by the optimize method
        self.v = self.gamma * self.v + self.alpha * gradient
        
        # 3. Update position
        x_next = x - self.v
        
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
        Runs the iterative Momentum Gradient Descent process.

        Args:
            x (ndarray): The initial starting point.
            func_callback (Callable): (Unused) A function that takes `x` and returns the loss.
            grad_func_callback (Callable): A function that takes `x` and returns the gradient.
            hessian_func_callback (Callable): (Unused)
            is_plot (bool): If True, returns the history of points.

        Returns:
            The final optimized solution `x` or (solution, history).
        """
        plot_points: list[ndarray] = [x]
        
        # [cite_start]Initialize velocity vector v(0) to zeros [cite: 2630, 2650]
        if self.v is None:
            self.v = np.zeros_like(x)

        # Stop when the gradient vector is close to zero
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
# 2. Nesterov Accelerated Gradient (NGD)
# =============================================================================

class NesterovGradientDescent(Optim):
    """
    Implementation of Nesterov Accelerated Gradient (NGD).

    This optimizer improves upon Momentum by calculating the gradient
    at a "look-ahead" position (where the velocity is about to take it)
    rather than at the current position. This "corrects" the
    [cite_start]velocity vector before the update [cite: 2668-2675].

    Args:
        alpha (float): The learning rate (e.g., 0.01).
        gamma (float): The momentum coefficient (e.g., 0.9).
    """

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.v: ndarray | None = None  # Velocity vector

        # Store initial values for reset
        self.alpha_init = alpha
        self.gamma_init = gamma
        return

    def _reset(self) -> None:
        """Resets the optimizer's state (iteration count and velocity)."""
        super()._reset()
        self.v = None
        self.alpha = self.alpha_init
        self.gamma = self.gamma_init
        return

    def _next(
        self, 
        x: ndarray, 
        grad_func_callback: Callable[[ndarray], ndarray]
    ) -> ndarray:
        """
        Calculates the next position using the NGD update equations.
        
        1. [cite_start]x_lookahead = x(k) - gamma * v(k-1) [cite: 2690]
        2. [cite_start]g_lookahead = grad(x_lookahead) [cite: 2691]
        3. [cite_start]v(k) = gamma * v(k-1) + alpha * g_lookahead [cite: 2692]
        4. [cite_start]x(k+1) = x(k) - v(k) [cite: 2693]
        """
        
        # 1. Compute look-ahead position
        # We assume self.v has been initialized by the optimize method
        x_lookahead = x - self.gamma * self.v
        
        # 2. Compute gradient at look-ahead position
        grad_lookahead = grad_func_callback(x_lookahead)
        
        # 3. Update velocity
        self.v = self.gamma * self.v + self.alpha * grad_lookahead
        
        # 4. Update position
        x_next = x - self.v
        
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
        Runs the iterative Nesterov Accelerated Gradient process.

        Args:
            x (ndarray): The initial starting point.
            func_callback (Callable): (Unused) A function that takes `x` and returns the loss.
            grad_func_callback (Callable): A function that takes `x` and returns the gradient.
            hessian_func_callback (Callable): (Unused)
            is_plot (bool): If True, returns the history of points.

        Returns:
            The final optimized solution `x` or (solution, history).
        """
        plot_points: list[ndarray] = [x]
        
        # [cite_start]Initialize velocity vector v(0) to zeros [cite: 2684]
        if self.v is None:
            self.v = np.zeros_like(x)

        # Stop when the gradient vector is close to zero
        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1
            x = self._next(x, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x