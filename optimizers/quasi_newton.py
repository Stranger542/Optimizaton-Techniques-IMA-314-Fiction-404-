import numpy as np
from numpy import ndarray
from typing import Callable, List,Tuple
from collections import deque
from utils.base import Optim, EPS

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
        self.H: ndarray | None = None 
        return

    def _reset(self) -> None:
        super()._reset()
        self.H = None
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
        plot_points: list[ndarray] = [x]

        if self.H is None:
            self.H = np.identity(len(x))
        g = grad_func_callback(x)
        while np.linalg.norm(g) > EPS:
            self.num_iter += 1
            p = -self.H @ g
            one_dim_func = lambda a_vec: func_callback(x + a_vec[0] * p)
            one_dim_grad = lambda a_vec: np.array([
                grad_func_callback(x + a_vec[0] * p) @ p
            ])
            
            alpha_vec = self.line_search.optimize(
                x=np.array([1.0]), 
                func_callback=one_dim_func,
                grad_func_callback=one_dim_grad,
                hessian_func_callback=lambda a: np.array([[1.0]]), 
                is_plot=False
            )
            alpha = alpha_vec[0]
            x_next = x + alpha * p
            delta = x_next - x
            g_next = grad_func_callback(x_next)
            y = g_next - g
            y_T_delta = y @ delta
            if y_T_delta > EPS:
                rho = 1.0 / y_T_delta
                I = np.identity(len(x))
                term1 = (I - rho * np.outer(delta, y))
                term2 = (I - rho * np.outer(y, delta))
                term3 = (rho * np.outer(delta, delta))
                self.H = term1 @ self.H @ term2 + term3
            else:
                pass 
            x = x_next
            g = g_next
            if is_plot:
                plot_points.append(x)
            if self.num_iter > 2000:
                print(f"Terminating: {self.__class__.__name__} reached max iterations.")
                break

        self._reset()
        if is_plot:
            return x, plot_points
        return x

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
        self.m = m  
        self.delta_history: deque[ndarray] = deque(maxlen=m)
        self.y_history: deque[ndarray] = deque(maxlen=m)
        return

    def _reset(self) -> None:
        super()._reset()
        self.delta_history.clear()
        self.y_history.clear()
        return

    def _get_search_direction(self, g: ndarray) -> ndarray:
        q = g.copy()
        alphas = []
        rhos = []
        for delta, y in reversed(list(zip(self.delta_history, self.y_history))):
            rho = 1.0 / (y @ delta)
            rhos.append(rho)
            alpha = rho * (delta @ q)
            alphas.append(alpha)
            q = q - alpha * y
        if len(self.delta_history) > 0:
            delta_k_1 = self.delta_history[-1]
            y_k_1 = self.y_history[-1]
            gamma = (delta_k_1 @ y_k_1) / (y_k_1 @ y_k_1)
            H0 = gamma * np.identity(len(g))
        else:
            H0 = np.identity(len(g))
        r = H0 @ q
        history_pairs = list(zip(self.delta_history, self.y_history))
        for (delta, y), rho, alpha in zip(history_pairs, reversed(rhos), reversed(alphas)):
            beta = rho * (y @ r)
            r = r + delta * (alpha - beta)
        return -r
    
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
        plot_points: list[ndarray] = [x]
        g = grad_func_callback(x)
        while np.linalg.norm(g) > EPS:
            self.num_iter += 1
            p = self._get_search_direction(g)
            one_dim_func = lambda a_vec: func_callback(x + a_vec[0] * p)
            one_dim_grad = lambda a_vec: np.array([
                grad_func_callback(x + a_vec[0] * p) @ p
            ])
            alpha_vec = self.line_search.optimize(
                x=np.array([1.0]), 
                func_callback=one_dim_func,
                grad_func_callback=one_dim_grad,
                hessian_func_callback=lambda a: np.array([[1.0]]), 
                is_plot=False
            )
            alpha = alpha_vec[0]
            x_next = x + alpha * p
            g_next = grad_func_callback(x_next)
            #Compute delta(k) and y(k)
            delta = x_next - x
            y = g_next - g
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