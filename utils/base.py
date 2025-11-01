import numpy as np
from numpy import ndarray
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from typing import Callable, list, tuple, Any

EPS = 1e-6

class Optim(ABC):
    """Abstract Base Class for all optimization algorithms (Optimizers).
    Any new optimizer (e.g., GradientDescent, Adam) should inherit
    from this class and implement the `optimize` method.
    """
    def __init__(self) -> None:
        self.num_iter: int = 0

    @abstractmethod
    def optimize(
        self,
        x: ndarray,
        func_callback: Callable[[ndarray], float],
        grad_func_callback: Callable[[ndarray], ndarray],
        hessian_func_callback: Callable[[ndarray], ndarray],
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        """
        Runs the optimization algorithm.
        Args:
            x (ndarray): The initial starting point (e.g., initial weights).
            func_callback (Callable): A function that takes `x` and returns the loss.
            grad_func_callback (Callable): A function that takes `x` and returns the gradient.
            hessian_func_callback (Callable): A function that takes `x` and returns the Hessian.
            is_plot (bool): If True, the optimizer should return the history of points.
        Returns:
            ndarray | tuple[ndarray, list[ndarray]]: 
            - The final optimized solution `x`.
            - If `is_plot` is True, returns a tuple of (solution, history_of_points).
        """
        pass

    def _reset(self) -> None:
        self.num_iter = 0
        return

class Function:
    """
    A wrapper class for mathematical functions.
    This class handles:
    1.  Evaluating the function.
    2.  Calculating the gradient (analytically if provided, otherwise numerically).
    3.  Calculating the Hessian (analytically if provided, otherwise numerically).
    4.  Interfacing with an `Optim` object to be optimized.
    5.  Plotting the function's 3D surface and an optimizer's trajectory.
    """

    def __init__(
        self, 
        func: Callable[[ndarray], float], 
        grad_func: Callable[[ndarray], ndarray] | None = None, 
        hessian_func: Callable[[ndarray], ndarray] | None = None, 
        name: str = "myFunc"
    ) -> None:
        """
        Args:
            func (Callable): The objective function, e.g., lambda x: x[0]**2 + x[1]**2.
            grad_func (Callable, optional): An analytical gradient function.
            hessian_func (Callable, optional): An analytical Hessian function.
            name (str, optional): A name for the function for plotting.
        """
        self.__func = func
        self.__grad_func = grad_func
        self.__hessian_func = hessian_func
        self.__name = name
        return None

    def __call__(self, x: ndarray) -> float:
        """Evaluates the function at point `x`."""
        return self.__func(x)

    def __repr__(self) -> str:
        """Returns the name of the function."""
        return self.__name

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient at point `x`.
        
        Uses the analytical gradient if provided, otherwise falls back
        to a 2D numerical approximation (central difference).
        """
        if self.__grad_func:
            return self.__grad_func(x)
        _x, _y = x
        df_dx = (
            self.__func(np.array([_x + EPS, _y]))
            - self.__func(np.array([_x - EPS, _y]))
        ) / (2 * EPS)
        df_dy = (
            self.__func(np.array([_x, _y + EPS]))
            - self.__func(np.array([_x, _y - EPS]))
        ) / (2 * EPS)
        return np.array([df_dx, df_dy])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the Hessian matrix at point `x`.
        
        Uses the analytical Hessian if provided, otherwise falls back
        to a 2D numerical approximation.
        """
        if self.__hessian_func:
            return self.__hessian_func(x)
        _x, _y = x
        d2f_dx2 = (
            self.__func(np.array([_x + EPS, _y]))
            - 2 * self.__func(np.array([_x, _y]))
            + self.__func(np.array([_x - EPS, _y]))
        ) / (EPS**2)
        d2f_dy2 = (
            self.__func(np.array([_x, _y + EPS]))
            - 2 * self.__func(np.array([_x, _y]))
            + self.__func(np.array([_x, _y - EPS]))
        ) / (EPS**2)

        d2f_dxdy = (
            self.__func(np.array([_x + EPS, _y + EPS]))
            - self.__func(np.array([_x + EPS, _y - EPS]))
            - self.__func(np.array([_x - EPS, _y + EPS]))
            + self.__func(np.array([_x - EPS, _y - EPS]))
        ) / (4 * EPS**2)

        hessian_matrix = np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])
        return hessian_matrix

    def optimize(
        self, initial_val: ndarray, optim: Optim, is_plot: bool = False
    ) -> ndarray:
        """
        Optimizes this function using a given optimizer.
        Args:
            initial_val (ndarray): The starting point for the optimization.
            optim (Optim): An instance of an optimizer (e.g., GradientDescent).
            is_plot (bool): Whether to plot the optimization trajectory.
        Returns:
            ndarray: The optimized solution.
        """
        soln = optim.optimize(
            initial_val, self.__call__, self.grad, self.hessian, is_plot=is_plot
        )
        if is_plot and isinstance(soln, tuple):
            self.plot(points=soln[1])
            assert isinstance(soln[0], ndarray)
            return soln[0] 
        assert isinstance(soln, ndarray), "Value received from Optim is corrupted"
        return soln

    def plot(
        self,
        points: list[ndarray] | ndarray | None = None,
        x_range: tuple[float, float] = (-10, 10),
        y_range: tuple[float, float] = (-10, 10),
        num_points: int = 100,
        show: bool = True,
    ) -> None | matplotlib.figure.Figure:
        """
        Plots a 3D surface of the function and optionally an 
        optimizer's trajectory.
        Args:
            points (list[ndarray], optional): A list of points in the trajectory.
            x_range (tuple, optional): The x-axis range to plot.
            y_range (tuple, optional): The y-axis range to plot.
            num_points (int, optional): The grid resolution for the surface.
            show (bool, optional): If True, shows the plot immediately.
        """

        if points is not None and len(points) > 0:
            points_array = np.array(points)
            x_range = (np.min(points_array[:, 0]) - 1, np.max(points_array[:, 0]) + 1)
            y_range = (np.min(points_array[:, 1]) - 1, np.max(points_array[:, 1]) + 1)

        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        try:
            Z = self.__call__(np.array([X, Y]))
        except ValueError:
            Z = np.array(
                [
                    self.__call__(np.array([xi, yi]))
                    for xi, yi in zip(X.flatten(), Y.flatten())
                ]
            ).reshape(X.shape)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="cividis",
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            alpha=0.8,
        )
        if points is not None and len(points) > 0:
            x_points = np.array([p[0] for p in points])
            y_points = np.array([p[1] for p in points])
            z_points = np.array([self.__call__(p) for p in points]) + 1e-3 

            ax.plot(
                x_points,
                y_points,
                z_points,
                color="r",
                marker="o",
                label="Trajectory",
                markersize=5,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_E_zlabel("Z")
        ax.set_title(self.__repr__())

        fig.colorbar(surf, shrink=0.5, aspect=5)
        if show:
            plt.show()
        else:
            return fig


class Algo(ABC):
    """
    Abstract Base Class for Machine Learning Algorithms
    that use an optimizer (e.g., Linear Regression).
    This class uses *composition*, holding an instance of an
    `Optim` object to perform its training.
    """
    @abstractmethod
    def __init__(self, optim: Optim, *args, **kwargs) -> None:
        """
        Args:
            optim (Optim): An optimizer instance (e.g., GradientDescent).
        """
        pass

    @abstractmethod
    def train(
        self, X_train: ndarray, Y_train: ndarray, epochs: int = 1, is_plot: bool = False
    ) -> None:
        """Trains the model on the given data."""
        pass

    @abstractmethod
    def test(self, X_test: ndarray, Y_test: ndarray, is_plot: bool) -> np.float32:
        """Tests the model and returns a performance metric (e.g., loss)."""
        pass

    @abstractmethod
    def __call__(self, X: ndarray, is_plot: bool = False) -> ndarray:
        """Makes predictions on new data `X`."""
        pass