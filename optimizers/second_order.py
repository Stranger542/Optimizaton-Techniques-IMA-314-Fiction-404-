import numpy as np
from numpy import ndarray
from typing import Callable
from utils.base import Optim, EPS

class NewtonMethodBase(Optim):

    def __init__(self, line_search: Optim | None = None) -> None:
        super().__init__()
        self.line_search = line_search
        return
    def _reset(self) -> None:
        super()._reset()
        return
    def _get_search_direction(self, g: ndarray, H: ndarray) -> ndarray:

        raise NotImplementedError("Child class must implement _get_search_direction")

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray],
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1
            g = grad_func_callback(x)
            H = hessian_func_callback(x)
            delta = self._get_search_direction(g, H)
            if np.allclose(delta, 0):
                print(f"Terminating: {self.__class__.__name__} stalled.")
                break
            if self.line_search:
                one_dim_func = lambda a_vec: func_callback(x + a_vec[0] * delta)
                one_dim_grad = lambda a_vec: np.array([
                    grad_func_callback(x + a_vec[0] * delta) @ delta
                ])
                alpha_vec = self.line_search.optimize(
                    x=np.array([1.0]),
                    func_callback=one_dim_func,
                    grad_func_callback=one_dim_grad,
                    hessian_func_callback=lambda a: np.array([[1.0]]), 
                    is_plot=False
                )
                alpha = alpha_vec[0]
            else:
                alpha = 1.0
            x = x + alpha * delta
            if is_plot:
                plot_points.append(x)            
            # Safety break
            if self.num_iter > 200:
                print(f"Terminating: {self.__class__.__name__} reached max iterations.")
                break
        self._reset()
        if is_plot:
            return x, plot_points
        return x

class NewtonMethod(NewtonMethodBase):


    def __init__(self, line_search: Optim | None = None) -> None:
        super().__init__(line_search=line_search)

    def _get_search_direction(self, g: ndarray, H: ndarray) -> ndarray:
        try:
            delta = np.linalg.solve(H, -g)
            return delta
        except np.linalg.LinAlgError:
            print("Warning: Hessian is singular. Newton's method failed to find "
                  "a direction. Halting.")
            return np.zeros_like(g) 

class DampedNewtonMethod(NewtonMethodBase):

    def __init__(self, line_search: Optim | None = None, damping_offset: float = 1e-4) -> None:
        super().__init__(line_search=line_search)
        self.damping_offset = damping_offset

    def _get_search_direction(self, g: ndarray, H: ndarray) -> ndarray:
        I = np.identity(H.shape[0])
        lambda_damping = 0.0

        try:
            eigenvals = np.linalg.eigvalsh(H)
            min_eig = np.min(eigenvals)
            if min_eig <= EPS:
                lambda_damping = abs(min_eig) + self.damping_offset
        except np.linalg.LinAlgError:
            print("Warning: Eigenvalue computation failed. Applying default damping.")
            lambda_damping = self.damping_offset
        H_damped = H + lambda_damping * I
        try:
            delta = np.linalg.solve(H_damped, -g)
            return delta
        except np.linalg.LinAlgError:
            print("Warning: Damped Hessian is singular. Falling back to gradient step.")
            return -g