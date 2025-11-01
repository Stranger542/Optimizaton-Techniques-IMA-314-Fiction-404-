
import numpy as np
from numpy import ndarray
from typing import Callable, List, Tuple
from utils.base import Optim, EPS

class BacktrackingLineSearch(Optim):
    """
    Implementation of Backtracking Line Search using the Armijo Condition.
    This algorithm is not a standalone optimizer, but a subroutine used
    by other optimizers (like GradientDescent, Newton, BFGS) to determine
    an appropriate step size `alpha` at each iteration.
    It finds an `alpha` that satisfies the sufficient decrease condition
    (Armijo rule) [cite: 1761, 1776, 1807-1808]:
        f(x + alpha * d) <= f(x) + c1 * alpha * (grad(x).T @ d)

    Args:
        alpha_init (float): The initial (largest) step size to try[cite: 1795].
        beta (float): The shrinking factor to reduce alpha (0 < beta < 1)[cite: 1796].
        c1 (float): The constant for the Armijo condition (0 < c1 < 1)[cite: 1796].
    """

    def __init__(self, alpha_init: float = 1.0, beta: float = 0.5, c1: float = 1e-4) -> None:
        super().__init__()
        self.alpha_init = alpha_init  # Initial step size to try
        self.beta = beta              # Shrinking factor
        self.c1 = c1                  # Armijo condition constant
    def _reset(self) -> None:
        super()._reset()
        return

    def _next(self, *args, **kwargs) -> ndarray:
        pass

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray] | None = None,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        """
        Performs the backtracking line search to find a valid step size alpha.
        This method is designed to be called by another optimizer.
        Args:
            x (ndarray): A 1D array containing the initial alpha, `[alpha_init]`.
            func_callback (Callable): The 1D objective function, phi(alpha).
                                    phi(a) = f(x_k + a * d_k)
            grad_func_callback (Callable): A callable that returns the 1D
                                         gradient, phi'(alpha).
                                         phi'(a) = grad(x_k + a * d_k).T @ d_k
            hessian_func_callback (Callable): (Unused)
            is_plot (bool): (Unused, returns no history)
        Returns:
            ndarray: A 1D array `[alpha]` containing the step size that
                     satisfies the Armijo condition.
        """
        alpha = x[0]
        phi_0 = func_callback(np.array([0.0]))
        phi_prime_0 = grad_func_callback(np.array([0.0]))[0]
        while func_callback(np.array([alpha])) > (phi_0 + self.c1 * alpha * phi_prime_0):
            # Shrink alpha
            alpha = self.beta * alpha
            self.num_iter += 1
            if alpha < 1e-12:
                break

        self._reset()
        final_alpha = np.array([alpha])
        
        if is_plot:
            return final_alpha, [final_alpha] 
            
        return final_alpha


class WolfeLineSearch(Optim):
    """
    Placeholder for a line search using the Wolfe Conditions.
    The Wolfe conditions [cite: 1738] combine the Armijo rule (sufficient decrease)
    with a curvature condition (the second Wolfe condition) to ensure
    the step size is not too small.
    This is more complex to implement and is not fully detailed in the
    provided lecture slides, so it is left as a placeholder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        print(f"Warning: {self.__class__.__name__} is not implemented.")
        pass

    def _reset(self) -> None:
        super()._reset()
        return

    def _next(self, *args, **kwargs) -> ndarray:
        pass

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray] | None = None,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        
        print(f"Warning: {self.__class__.__name__} is not implemented. "
              f"Returning initial alpha.")
        
        if is_plot:
            return x, [x]
        return x