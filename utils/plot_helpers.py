import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import List, Callable, Dict, Any

def plot_loss_curves(
    histories: Dict[str, List[ndarray]], 
    loss_func_callable: Callable[..., float],
    is_log_scale: bool = True
):
    """
    Plots the loss (objective function value) over iterations for
    multiple optimizers.
    This is ideal for comparing the convergence speed of optimizers
    like GD, SGD, and Mini-Batch GD.
    Args:
        histories (Dict): A dictionary where keys are optimizer names
                          (e.g., "GD", "SGD") and values are the list
                          of weight vectors (the path) from their history.
        loss_func_callable (Callable): The function to call to compute the
                                     loss at each point in the history.
                                     (e.g., a lambda W: loss_func(W, X, Y))
        is_log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    for name, path in histories.items():
        try:
            loss_history = [loss_func_callable(W) for W in path]
        except Exception as e:
            print(f"Error computing loss for {name}: {e}")
            print("Ensure your loss function can accept a single weight vector.")
            continue
        plt.plot(loss_history, label=name, alpha=0.8)
    
    plt.xlabel("Iteration / Step")
    plt.ylabel("Loss f(x)" + (" (Log Scale)" if is_log_scale else ""))
    plt.title("Optimizer Loss Convergence")
    plt.legend()
    if is_log_scale:
        plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_contour_comparison(
    func_callable: Callable[[ndarray], float],
    histories: Dict[str, List[ndarray]], 
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
    title: str = "Optimizer Path Comparison"
):
    """
    Creates a 2D contour plot of a function and plots the
    optimization paths for multiple optimizers on the same graph.
    Args:
        func_callable (Callable): The 2D function to plot (e.g., Rosenbrock.func).
        histories (Dict): A dictionary where keys are optimizer names
                          (e.g., "Momentum", "Adam") and values are
                          the list of points [x, y] in their path.
        x_range (tuple): The plotting range for the x-axis.
        y_range (tuple): The plotting range for the y-axis.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 9))
    x_lin = np.linspace(x_range[0], x_range[1], 100)
    y_lin = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_lin, y_lin)
    Z = np.array([
        func_callable(np.array([xi, yi]))
        for xi, yi in zip(X.flatten(), Y.flatten())
    ]).reshape(X.shape)
    plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap="cividis", alpha=0.7)
    for name, path in histories.items():
        path_arr = np.array(path)
        plt.plot(path_arr[:, 0], path_arr[:, 1], '-o', 
                 label=name, 
                 markersize=3, alpha=0.7, linewidth=2)
    if histories:
        first_path = np.array(list(histories.values())[0])
        plt.plot(first_path[0, 0], first_path[0, 1], 'go', 
                 markersize=12, label='Start', zorder=5)
    
    plt.xlabel("x1 (w1)")
    plt.ylabel("x2 (w2)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=0.5, zorder=1)
    plt.axvline(0, color='black', linewidth=0.5, zorder=1)
    plt.show()