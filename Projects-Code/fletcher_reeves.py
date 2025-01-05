import numpy as np

def fletcher_reeves_method(func, grad, x0, tol=1e-5, max_iter=100, console_output=None, restart_interval=10):
    x = x0
    g = grad(x)
    d = -g
    history = [x.copy()]
    prev_fval = func(x)

    for i in range(max_iter):
        alpha = -np.dot(g, d) / np.dot(d, np.dot(np.eye(len(x)), d))
        x_new = x + alpha * d

        fval = func(x_new)
        if np.linalg.norm(grad(x_new)) < tol:
            if console_output is not None:
                console_output.append(f"{i+1}. iterasyonda gradyan normu ile yakınsama sağlandı.")
            break
        if abs(fval - prev_fval) < tol or np.linalg.norm(x_new - x) < tol:
            if console_output is not None:
                console_output.append(f"{i+1}. iterasyonda yakınsama sağlandı.")
            break

        if i > 0 and i % restart_interval == 0:
            d = -grad(x_new)
            if console_output is not None:
                console_output.append(f"{i+1}. iterasyonda yeniden başlatma yapıldı.")
        else:
            g_new = grad(x_new)
            beta = np.dot(g_new, g_new) / np.dot(g, g)
            d = -g_new + beta * d
            g = g_new

        prev_fval = fval
        x = x_new
        history.append(x.copy())

        # Text alanını güncellemek için çağrı
        if console_output is not None:
            console_output.append(f"Iterasyon {i+1}: x = {x}, f(x) = {fval}")
            # Dışarıdan gelen update_console fonksiyonu kullanılabilir

    return x, history
