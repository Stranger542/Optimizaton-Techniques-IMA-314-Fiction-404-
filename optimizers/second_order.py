# optimizers/second_order.py

import numpy as np
from numpy import ndarray
from typing import Callable, list, tuple

# Import the base classes from your utils folder
from utils.base import Optim, EPS

# =============================================================================
# 1. Base Class for Newton's Methods
# =============================================================================

class NewtonMethodBase(Optim):
    """
    A base class for Newton's optimization methods.

    This class handles the main optimization loop, including an
    optional line search to find the step size `alpha`. The core
    logic of *how* the search direction `delta` is computed is
    left to the child classes (Pure vs. Damped).

    Args:
        line_search (Optim, optional): An optimizer (like BacktrackingLineSearch)
            to find the optimal step size `alpha` at each iteration.
            If None, a "pure" step (`alpha=1.0`) is always used.
    """

    def __init__(self, line_search: Optim | None = None) -> None:
        super().__init__()
        self.line_search = line_search
        return

    def _reset(self) -> None:
        """Resets the iteration count."""
        super()._reset()
        return

    def _get_search_direction(self, g: ndarray, H: ndarray) -> ndarray:
        """
        Abstract method to compute the search direction `delta`.
        This is where child classes implement their logic.
        
        Args:
            g (ndarray): The current gradient vector.
            H (ndarray): The current Hessian matrix.

        Returns:
            ndarray: The search direction `delta`.
        """
        raise NotImplementedError("Child class must implement _get_search_direction")

    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray],
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        """
        Runs the iterative Newton's method process.
        """
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPS:
            self.num_iter += 1

            # 1. Get current gradient and Hessian
            g = grad_func_callback(x)
            H = hessian_func_callback(x)

            # 2. Compute the search direction (delegated to child class)
            # This is the step delta that solves H * delta = -g
            delta = self._get_search_direction(g, H)

            # 3. Check for stall or failure
            if np.allclose(delta, 0):
                print(f"Terminating: {self.__class__.__name__} stalled.")
                break

            # 4. Perform line search to find step size `alpha`
            if self.line_search:
                # We need to find an `alpha` that minimizes phi(alpha) = f(x + alpha * delta)
                # We create 1D functions for the line search optimizer.
                
                # phi(a) = f(x + a * d)
                one_dim_func = lambda a_vec: func_callback(x + a_vec[0] * delta)
                
                # phi'(a) = grad(x + a * d).T @ d (Chain rule)
                one_dim_grad = lambda a_vec: np.array([
                    grad_func_callback(x + a_vec[0] * delta) @ delta
                ])

                # Run the line search (e.g., Backtracking)
                # We start the search from alpha = 1.0
                alpha_vec = self.line_search.optimize(
                    x=np.array([1.0]),
                    func_callback=one_dim_func,
                    grad_func_callback=one_dim_grad,
                    hessian_func_callback=lambda a: np.array([[1.0]]), # Placeholder
                    is_plot=False
                )
                alpha = alpha_vec[0]
            else:
                # Use the "pure" Newton step
                alpha = 1.0

            # 5. Update the position
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


# =============================================================================
# 2. Standard (Pure) Newton's Method
# =============================================================================

class NewtonMethod(NewtonMethodBase):
    """
    Implementation of the standard "pure" Newton's Method.
    
    This method computes the search direction `delta` by solving
    the linear system `H * delta = -g` directly, where `H` is the
    exact Hessian.
    
    This method is very fast (quadratic convergence) near the
    optimum but can be unstable if the Hessian is not
    positive definite (i.e., not convex)[cite: 1990].
    """

    def __init__(self, line_search: Optim | None = None) -> None:
        super().__init__(line_search=line_search)

    def _get_search_direction(self, g: ndarray, H: ndarray) -> ndarray:
        """
        Solves the pure Newton system `H * delta = -g`.
        """
        try:
            # Solve the linear system
            delta = np.linalg.solve(H, -g)
            return delta
        except np.linalg.LinAlgError:
            # This happens if H is singular (e.g., non-invertible)
            print("Warning: Hessian is singular. Newton's method failed to find "
                  "a direction. Halting.")
            return np.zeros_like(g) # Return a zero vector to stall


# =============================================================================
# 3. Damped Newton's Method
# =============================================================================

class DampedNewtonMethod(NewtonMethodBase):
    """
    Implementation of the Damped Newton's Method.

    This is a modification to make Newton's method more robust
    when the Hessian `H` is not positive definite[cite: 1990, 2014].
    
    It solves a modified system: `(H + lambda*I) * delta = -g`,
    where `lambda` is a "damping factor" [cite: 2014] that ensures
    the modified Hessian is positive definite and well-conditioned.

    Args:
        line_search (Optim, optional): An optimizer for line search.
        damping_offset (float): A small positive value to add to the
            damping factor to ensure strict positive definiteness.
    """

    def __init__(self, line_search: Optim | None = None, damping_offset: float = 1e-4) -> None:
        super().__init__(line_search=line_search)
        self.damping_offset = damping_offset

    def _get_search_direction(self, g: ndarray, H: ndarray) -> ndarray:
        """
        Solves the damped Newton system `(H + lambda*I) * delta = -g`.
        
        The damping factor `lambda` is chosen dynamically to be just
        large enough to make the Hessian positive definite.
        """
        I = np.identity(H.shape[0])
        lambda_damping = 0.0

        try:
            # 1. Check if H is positive definite by checking its eigenvalues
            # We use eigvalsh because the Hessian is symmetric
            eigenvals = np.linalg.eigvalsh(H)
            min_eig = np.min(eigenvals)

            if min_eig <= EPS:
                # H is not positive definite [cite: 1990]
                # We must add damping
                # We shift the eigenvalues so the new minimum is `damping_offset`
                lambda_damping = abs(min_eig) + self.damping_offset
        
        except np.linalg.LinAlgError:
            # Failed to compute eigenvalues, H is likely ill-conditioned
            # Apply a default damping
            print("Warning: Eigenvalue computation failed. Applying default damping.")
            lambda_damping = self.damping_offset

        # 2. Create the damped Hessian
        # H_modified = H + lambda*I [cite: 2019-2025]
        H_damped = H + lambda_damping * I

        # 3. Solve the damped system
        try:
            delta = np.linalg.solve(H_damped, -g)
            return delta
        except np.linalg.LinAlgError:
            # This should rarely happen now, but if it does, fall back
            # to a simple gradient step.
            print("Warning: Damped Hessian is singular. Falling back to gradient step.")
            return -g