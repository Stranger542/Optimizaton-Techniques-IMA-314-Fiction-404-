import numpy as np
from numpy import ndarray
from typing import Callable
from utils.base import Optim, EPS

class MomentumGradientDescent(Optim):

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.v: ndarray | None = None  
        self.alpha_init = alpha
        self.gamma_init = gamma
        return

    def _reset(self) -> None:
        super()._reset()
        self.v = None
        self.alpha = self.alpha_init
        self.gamma = self.gamma_init
        return

    def _next(
        self, 
        x: ndarray, 
        gradient: ndarray
    ) -> ndarray:
        # Initialize velocity if None
        if self.v is None:
            self.v = np.zeros_like(x)
        
        self.v = self.gamma * self.v + self.alpha * gradient
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
    
        self.num_iter = 0  # Reset iteration count at the beginning
        plot_points: list[ndarray] = [x.copy()]
        
        # Initialize velocity
        if self.v is None:
            self.v = np.zeros_like(x)
        
        g = grad_func_callback(x)  # Get initial gradient
        
        while np.linalg.norm(g) > EPS:
            self.num_iter += 1
            x = self._next(x, g)
            g = grad_func_callback(x)  # Get new gradient for next check

            if is_plot:
                plot_points.append(x.copy())
            
            if self.num_iter > 50000:
                print(f"Warning: {self.__class__.__name__} reached max iterations.")
                break

        # Store iter count before resetting state
        final_iter_count = self.num_iter
        self._reset()
        self.num_iter = final_iter_count  # Restore it for printing

        if is_plot:
            return x, plot_points
        return x


class NesterovGradientDescent(Optim):

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.v: ndarray | None = None  
        self.alpha_init = alpha
        self.gamma_init = gamma
        return

    def _reset(self) -> None:
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

        # Initialize velocity if None
        if self.v is None:
            self.v = np.zeros_like(x)
        
        # Compute look-ahead position
        x_lookahead = x - self.gamma * self.v
        # Compute gradient at look-ahead position
        grad_lookahead = grad_func_callback(x_lookahead)
        # Update velocity
        self.v = self.gamma * self.v + self.alpha * grad_lookahead
        # Update position
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

        self.num_iter = 0  # Reset iteration count at the beginning
        plot_points: list[ndarray] = [x.copy()]
        
        # Initialize velocity
        if self.v is None:
            self.v = np.zeros_like(x)
        
        g = grad_func_callback(x)  # Get initial gradient
        
        while np.linalg.norm(g) > EPS:
            self.num_iter += 1
            x = self._next(x, grad_func_callback)
            g = grad_func_callback(x)  # Get new gradient for next check

            if is_plot:
                plot_points.append(x.copy())
            
            if self.num_iter > 50000:
                print(f"Warning: {self.__class__.__name__} reached max iterations.")
                break

        # Store iter count before resetting state
        final_iter_count = self.num_iter
        self._reset()
        self.num_iter = final_iter_count  # Restore it for printing

        if is_plot:
            return x, plot_points
        return x