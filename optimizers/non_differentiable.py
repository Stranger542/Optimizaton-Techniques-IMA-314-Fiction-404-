import numpy as np
from numpy import ndarray
from typing import Callable, Literal
from utils.base import Optim, EPS

class SubGradientMethod(Optim):

    def __init__(
        self, 
        alpha: float = 0.01, 
        n_iterations: int = 1000, 
        policy: Literal['fixed', 'diminishing'] = 'diminishing'
    ) -> None:
        super().__init__()
        self.alpha_init = alpha
        self.n_iterations = n_iterations
        self.policy = policy
        return

    def _reset(self) -> None:
        super()._reset()
        return

    def _get_alpha(self, k: int) -> float:
        if self.policy == 'fixed':
            return self.alpha_init
        elif self.policy == 'diminishing':
            # Implements a diminishing step size, e.g., alpha_k = alpha_0 / (k + 1)
            return self.alpha_init / (k + 1)
        else:
            raise ValueError(f"Unknown step size policy: {self.policy}")

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
        x_best = x.copy()
        f_best = func_callback(x)

        for k in range(self.n_iterations):
            self.num_iter = k + 1 
            alpha_k = self._get_alpha(k)
            g_k = grad_func_callback(x)
            x = x - alpha_k * g_k
            f_new = func_callback(x)
            if f_new < f_best:
                f_best = f_new
                x_best = x.copy()

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x_best, plot_points
        return x_best