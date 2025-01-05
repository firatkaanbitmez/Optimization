import numpy as np
import matplotlib.pyplot as plt

def function(x):
    """
    Amaç fonksiyonu f(x1, x2) = x1^2 - x1*x2 + x2^2 + x1 + x2.
    """
    x1, x2 = x
    return x1**2 - x1*x2 + x2**2 + x1 + x2

def gradient(x):
    """
    Fonksiyonun gradyanı (türev vektörü).
    """
    x1, x2 = x
    return np.array([2*x1 - x2 + 1, 2*x2 - x1 + 1])

def hessian(_x):
    """
    Hessian matrisi hesaplanır.
    """
    return np.array([[2, -1], [-1, 2]])

def plot_optimization(func, history):
    """
    Optimizasyon sürecini görselleştirir.
    """
    x1_vals = np.linspace(-10, 10, 400)
    x2_vals = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = func([X1, X2])

    plt.figure()
    plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=5)
    plt.title('Optimization Path')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
