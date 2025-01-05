import numpy as np

def newtons_method(func, grad, hessian, x0, tol=1e-5, max_iter=100, console_output=None):
    """
    Newton'un yönteminin uygulanması.
    """
    x = x0
    history = [x.copy()]

    for i in range(max_iter):
        g = grad(x)
        h = hessian(x)

        if np.linalg.norm(g) < tol:
            if console_output is not None:
                console_output.append(f"{i+1}. iterasyonda gradyan normu ile yakınsama sağlandı.")
            break

        try:
            delta_x = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            if console_output is not None:
                console_output.append("Hessian matrisi tekil, çözüm başarısız oldu.")
            break

        x_new = x + delta_x
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return x, history
