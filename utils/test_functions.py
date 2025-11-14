import numpy as np
from numpy import ndarray
import random
from utils.base import Function

def quadratic_func(x: ndarray) -> float:
    """f(x, y) = x^2 + y^2"""
    return x[0]**2 + x[1]**2

def quadratic_grad(x: ndarray) -> ndarray:
    """Gradient of f(x, y) = [2x, 2y]"""
    return np.array([2 * x[0], 2 * x[1]])

def quadratic_hess(x: ndarray) -> ndarray:
    """Hessian of f(x, y) = [[2, 0], [0, 2]]"""
    return np.array([[2.0, 0.0], [0.0, 2.0]])

Quadratic = Function(
    func=quadratic_func,
    grad_func=quadratic_grad,
    hessian_func=quadratic_hess,
    name="Simple Quadratic Function (f(x,y) = x^2 + y^2)"
)

A_rosen = 1.0
B_rosen = 10.0

def rosenbrock_func(x: ndarray) -> float:
    """f(x, y) = (A - x)^2 + B * (y - x^2)^2"""
    return (A_rosen - x[0])**2 + B_rosen * (x[1] - x[0]**2)**2

def rosenbrock_grad(x: ndarray) -> ndarray:
    """Analytical gradient of the Rosenbrock function"""
    dx = -2.0 * (A_rosen - x[0]) - 4.0 * B_rosen * (x[1] - x[0]**2) * x[0]
    dy = 2.0 * B_rosen * (x[1] - x[0]**2)
    return np.array([dx, dy])

def rosenbrock_hess(x: ndarray) -> ndarray:
    """Analytical Hessian of the Rosenbrock function"""
    dxx = 2.0 - 4.0 * B_rosen * (x[1] - 3 * x[0]**2)
    dxy = -4.0 * B_rosen * x[0]
    dyy = 2.0 * B_rosen
    return np.array([[dxx, dxy], [dxy, dyy]])

Rosenbrock = Function(
    func=rosenbrock_func,
    grad_func=rosenbrock_grad,
    hessian_func=rosenbrock_hess,
    name="Rosenbrock Function (A=1, B=10)"
)

A_rat = 10
def rastrigin_func(xy: ndarray) -> float:
    return (
        A_rat * 2
        + (xy[0] ** 2 - A_rat * np.cos(2 * np.pi * xy[0]))
        + (xy[1] ** 2 - A_rat * np.cos(2 * np.pi * xy[1]))
    )

Rastrigin = Function(
    func=rastrigin_func,
    name="Rastrigin Function"
)


A_ack = 20
B_ack = 0.2
C_ack = 2 * np.pi
def ackley_func(xy: ndarray) -> float:
    return (
        -A_ack * np.exp(-B_ack * np.sqrt((xy[0] ** 2 + xy[1] ** 2) / 2))
        - np.exp((np.cos(C_ack * xy[0]) + np.cos(C_ack * xy[1])) / 2)
        + A_ack
        + np.exp(1)
    )

Ackley = Function(
    func=ackley_func,
    name="Ackley Function"
) 


def bohachevsky_func(xy: ndarray) -> float:
    return (
        xy[0] ** 2
        + 2 * xy[1] ** 2
        - 0.3 * np.cos(3 * np.pi * xy[0])
        - 0.4 * np.cos(4 * np.pi * xy[1])
        + 0.7
    )

Bohachevsky = Function(
    func=bohachevsky_func,
    name="Bohachevsky Function"
)

def trid_func(xy: ndarray) -> float:
    return (
        (xy[0] - 1) ** 2
        + (xy[1] - 1) ** 2
        - xy[0] * xy[1] 
    )

Trid = Function(
    func=trid_func,
    name="Trid Function"
) 


def rotated_hyper_ellipsoid_func(xy: ndarray) -> float:
    return xy[0] ** 2 + (xy[0] ** 2 + xy[1] ** 2)

RotatedHyperEllipsoid = Function(
    func=rotated_hyper_ellipsoid_func,
    name="Rotated Hyper-Ellipsoid"
) 


def piecewise_linear_func(x: ndarray) -> float:
    return np.abs(x[0]) + 2 * np.abs(x[1])

def piecewise_linear_subgrad(x: ndarray) -> ndarray:
    """
    The sub-differential at x=0 is the interval [-1, 1].
    The sub-differential at y=0 is the interval [-2, 2].
    """
    g1 = np.sign(x[0]) if x[0] != 0 else random.uniform(-1, 1)
    g2 = 2 * np.sign(x[1]) if x[1] != 0 else random.uniform(-2, 2)
    return np.array([g1, g2])

PiecewiseLinear = Function(
    func=piecewise_linear_func,
    grad_func=piecewise_linear_subgrad,
    hessian_func=lambda x: np.array([[0,0],[0,0]]), 
    name="Piecewise Linear (f = |x| + 2|y|)"
)


def generate_linear_regression_data(N: int = 100, d: int = 1):
    W_true = np.random.randn(d + 1, 1)
    X = np.random.rand(N, d)
    X_aug = np.hstack([np.ones((N, 1)), X])
    noise = 0.5 * np.random.randn(N, 1)
    Y = X_aug @ W_true + noise
    return X_aug, Y, W_true

def linear_regression_loss(W: ndarray, X: ndarray, Y: ndarray) -> float:
    N = X.shape[0]
    E = X @ W - Y
    loss = (1 / (2 * N)) * (E.T @ E)
    return loss.item()

def linear_regression_gradient(W: ndarray, X: ndarray, Y: ndarray) -> ndarray:
    N = X.shape[0]
    E = X @ W - Y
    gradient = (1 / N) * X.T @ E
    return gradient.flatten() 


def generate_logistic_regression_data(N: int = 200, d: int = 2):

    if d != 2:
        raise ValueError("This specific generator only supports d=2.")
        
    N_per_class = N // 2
    
    # Class 0: Centered at (-2, -2)
    X_0 = np.random.randn(N_per_class, d) + np.array([-2, -2])
    Y_0 = np.zeros((N_per_class, 1))
    
    # Class 1: Centered at (2, 2)
    X_1 = np.random.randn(N_per_class, d) + np.array([2, 2])
    Y_1 = np.ones((N_per_class, 1))
    
    # Concatenate and shuffle
    X = np.vstack([X_0, X_1])
    Y = np.vstack([Y_0, Y_1])
    
    permutation = np.random.permutation(N)
    
    return X[permutation], Y[permutation]