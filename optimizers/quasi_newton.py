# optimizers/quasi_newton.py

import numpy as np
from numpy import ndarray
from typing import Callable, list, tuple
from collections import deque

# Import the base classes from your utils folder
from utils.base import Optim, EPS

# =============================================================================
# 1. BFGS (Broyden, Fletcher, Goldfarb, Shanno)
# =============================================================================

class BFGS(Optim):
    """
    Implementation of the BFGS (Broyden, Fletcher, Goldfarb, Shanno) Algorithm.

    This is a quasi-Newton method that iteratively builds an approximation
    of the inverse Hessian matrix, H, using only gradient information.
    This avoids the computation and inversion of the true Hessian.

    Args:
        line_search (Optim): A line search optimizer (e.g., BacktrackingLineSearch)
            used to find the step size `alpha` at each iteration.
    """

    def __init__(self, line_search: Optim) -> None:
        super().__init__()
        self.line_search = line_search
        self.H: ndarray | None = None  # Inverse Hessian Approximation
        return

    def _reset(self) -> None:
        """Resets the optimizer's state."""
        super()._reset()
        self.H = None
        return
    
    def _next(self, *args, **kwargs) -> ndarray:
        """_next is not used; logic is in the `optimize` loop."""
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
        Runs the iterative BFGS optimization process.
        """
        plot_points: list[ndarray] = [x]
        
        # Initialize H(0) as the identity matrix [cite: 1884]
        if self.H is None:
            self.H = np.identity(len(x))
            
        g = grad_func_callback(x)

        while np.linalg.norm(g) > EPS:
            self.num_iter += 1

            # 1. Compute search direction p(k) = -H(k) * g(k) [cite: 1887]
            p = -self.H @ g

            # 2. Perform line search to find step size alpha(k) [cite: 1888]
            # We need to find an `alpha` that minimizes phi(alpha) = f(x + alpha * p)
            one_dim_func = lambda a_vec: func_callback(x + a_vec[0] * p)
            one_dim_grad = lambda a_vec: np.array([
                grad_func_callback(x + a_vec[0] * p) @ p
            ])
            
            alpha_vec = self.line_search.optimize(
                x=np.array([1.0]),  # Start with a full step
                func_callback=one_dim_func,
                grad_func_callback=one_dim_grad,
                hessian_func_callback=lambda a: np.array([[1.0]]), # Placeholder
                is_plot=False
            )
            alpha = alpha_vec[0]

            # 3. Update position x(k+1) [cite: 1889]
            x_next = x + alpha * p
            
            # 4. Compute delta(k) = x(k+1) - x(k) [cite: 1891]
            delta = x_next - x
            
            # 5. Compute new gradient g(k+1) [cite: 1890]
            g_next = grad_func_callback(x_next)
            
            # 6. Compute y(k) = g(k+1) - g(k) [cite: 1892]
            y = g_next - g
            
            # 7. Compute rho(k) = 1 / (y(k).T @ delta(k)) [cite: 1893]
            # Ensure curvature condition is met (y.T @ delta > 0) [cite: 1894]
            y_T_delta = y @ delta
            if y_T_delta > EPS:
                rho = 1.0 / y_T_delta
                
                # 8. Update H(k+1) using the BFGS formula [cite: 1895]
                I = np.identity(len(x))
                
                term1 = (I - rho * np.outer(delta, y))
                term2 = (I - rho * np.outer(y, delta))
                term3 = (rho * np.outer(delta, delta))
                
                self.H = term1 @ self.H @ term2 + term3
            else:
                # Curvature condition not met, skip update [cite: 1896]
                pass 
                
            # Update state for next iteration
            x = x_next
            g = g_next

            if is_plot:
                plot_points.append(x)
                
            # Safety break
            if self.num_iter > 2000:
                print(f"Terminating: {self.__class__.__name__} reached max iterations.")
                break

        self._reset()
        if is_plot:
            return x, plot_points
        return x


# =============================================================================
# 2. L-BFGS (Limited-memory BFGS)
# =============================================================================

class LBFGS(Optim):
    """
    Implementation of the Limited-memory BFGS (L-BFGS) Algorithm.

    L-BFGS is a quasi-Newton method that approximates the BFGS
    algorithm using only a limited amount of computer memory.
    It does not store the full n x n inverse Hessian approximation.
    
    Instead, it computes the search direction using the "two-loop
    recursion" based on a history of the last `m` {delta, y} pairs.
    
    Args:
        line_search (Optim): A line search optimizer (e.g., BacktrackingLineSearch).
        m (int): The memory size, i.e., the number of past {delta, y}
                 pairs to store.
    """
    def __init__(self, line_search: Optim, m: int = 10) -> None:
        super().__init__()
        self.line_search = line_search
        self.m = m  # Memory size [cite: 1980]
        
        # Use deques to efficiently store the history
        # They automatically discard old items when the max size is reached
        self.delta_history: deque[ndarray] = deque(maxlen=m)
        self.y_history: deque[ndarray] = deque(maxlen=m)
        
        return

    def _reset(self) -> None:
        """Resets the optimizer's state."""
        super()._reset()
        self.delta_history.clear()
        self.y_history.clear()
        return

    def _get_search_direction(self, g: ndarray) -> ndarray:
        """
        Computes the search direction p = -H * g using the
        L-BFGS two-loop recursion .
        """
        q = g.copy()
        
        # We will store the computed 'alpha' scalars from the first loop
        # to reuse them in the second loop
        alphas = []
        rhos = []
        
        # --- First Loop (Backward) [cite: 1979-1983] ---
        # Iterate from newest to oldest (i = k-1 to k-m)
        for delta, y in reversed(list(zip(self.delta_history, self.y_history))):
            rho = 1.0 / (y @ delta)
            rhos.append(rho)
            
            alpha = rho * (delta @ q)
            alphas.append(alpha)
            
            q = q - alpha * y

        # --- Initial Hessian Approximation H(0) ---
        # A common heuristic is to scale the identity matrix
        # This helps approximate the scale of the true Hessian [cite: 2000]
        if len(self.delta_history) > 0:
            delta_k_1 = self.delta_history[-1]
            y_k_1 = self.y_history[-1]
            gamma = (delta_k_1 @ y_k_1) / (y_k_1 @ y_k_1)
            H0 = gamma * np.identity(len(g))
        else:
            H0 = np.identity(len(g))

        r = H0 @ q
        
        # --- Second Loop (Forward) [cite: 1995-1997] ---
        # Iterate from oldest to newest (i = k-m to k-1)
        # We must reverse our stored lists (alphas, rhos) and history
        history_pairs = list(zip(self.delta_history, self.y_history))
        
        for (delta, y), rho, alpha in zip(history_pairs, reversed(rhos), reversed(alphas)):
            beta = rho * (y @ r)
            r = r + delta * (alpha - beta)
            
        # The final vector `r` is the search direction
        # We return p = -r
        return -r
    
    def _next(self, *args, **kwargs) -> ndarray:
        """_next is not used; logic is in the `optimize` loop."""
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
        Runs the iterative L-BFGS optimization process.
        """
        plot_points: list[ndarray] = [x]
        
        g = grad_func_callback(x)

        while np.linalg.norm(g) > EPS:
            self.num_iter += 1

            # 1. Compute search direction p(k) using two-loop recursion
            p = self._get_search_direction(g)
            
            # 2. Perform line search to find step size alpha(k) [cite: 2003]
            one_dim_func = lambda a_vec: func_callback(x + a_vec[0] * p)
            one_dim_grad = lambda a_vec: np.array([
                grad_func_callback(x + a_vec[0] * p) @ p
            ])
            
            alpha_vec = self.line_search.optimize(
                x=np.array([1.0]),  # Start with a full step
                func_callback=one_dim_func,
                grad_func_callback=one_dim_grad,
                hessian_func_callback=lambda a: np.array([[1.0]]), # Placeholder
                is_plot=False
            )
            alpha = alpha_vec[0]
            
            # 3. Update position x(k+1) [cite: 2004]
            x_next = x + alpha * p
            
            # 4. Compute new gradient g(k+1) [cite: 2011]
            g_next = grad_func_callback(x_next)
            
            # 5. Compute delta(k) and y(k)
            delta = x_next - x
            y = g_next - g
            
            # 6. Store {delta, y} in history [cite: 2007, 2008]
            if (y @ delta) > EPS: # Only store if curvature is positive
                self.delta_history.append(delta)
                self.y_history.append(y)
                
            # Update state for next iteration
            x = x_next
            g = g_next

            if is_plot:
                plot_points.append(x)
                
            # Safety break
            if self.num_iter > 2000:
                print(f"Terminating: {self.__class__.__name__} reached max iterations.")
                break
                
        self._reset()
        if is_plot:
            return x, plot_points
        return x