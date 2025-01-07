import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt

# Minimize edilecek fonksiyon ve türevini 
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
    df_dx1 = 2*x1 - x2 + 1  # x1'e göre türev
    df_dx2 = 2*x2 - x1 + 1  # x2'ye göre türev
    return np.array([df_dx1, df_dx2])

def hessian(_x):
    """
    Hessian matrisi hesaplanır.
    """
    return np.array([[2, -1], [-1, 2]])

def fletcher_reeves_method(func, grad, x0, tol=1e-5, max_iter=100, console_output=None, restart_interval=10):
    x = x0
    g = grad(x)
    
    if np.linalg.norm(g) < tol:
        if console_output is not None:
            console_output.append("Başlangıç noktasının gradyanı çok küçük, doğrudan çözüm kabul edildi.")
        return x, [x.copy()]

    d = -g
    history = [x.copy()]
    prev_fval = func(x)

    for i in range(max_iter):
        if np.linalg.norm(d) < 1e-15:
            if console_output is not None:
                console_output.append(f"{i+1}. iterasyonda d vektörü ~ 0, algoritma durduruldu.")
            break
        
        # Adım büyüklüğü
        alpha = -np.dot(g, d) / np.dot(d, d)  
        
        x_new = x + alpha * d
        fval = func(x_new)

        # Yakınsama kontrolleri
        if np.linalg.norm(grad(x_new)) < tol:
            msg = f"{i+1}. iterasyonda gradyan normu ile yakınsama sağlandı."
            if console_output is not None:
                console_output.append(msg)
            break
        if abs(fval - prev_fval) < tol:
            msg = f"{i+1}. iterasyonda fonksiyon değeri değişimi ile yakınsama sağlandı."
            if console_output is not None:
                console_output.append(msg)
            break
        if np.linalg.norm(x_new - x) < tol:
            msg = f"{i+1}. iterasyonda vektör değişimi ile yakınsama sağlandı."
            if console_output is not None:
                console_output.append(msg)
            break

        # Yeniden başlatma
        if i > 0 and i % restart_interval == 0:
            d = -grad(x_new)
            beta = 0.0  # restart sırasında beta'yı 0 kabul
            msg = f"{i+1}. iterasyonda yeniden başlatma yapıldı (beta=0)."
            if console_output is not None:
                console_output.append(msg)
        else:
            g_new = grad(x_new)
            beta = np.dot(g_new, g_new) / np.dot(g, g)
            d = -g_new + beta * d
            g = g_new

        x = x_new
        prev_fval = fval
        history.append(x.copy())
        
        # Bilgi mesajı
        msg = (f"Iterasyon {i+1}:\n"
               f"    x = {x}\n"
               f"    f(x) = {fval}\n"
               f"    Gradyan = {g}\n"
               f"    Beta = {beta}\n"
               f"    Adım Büyüklüğü = {alpha}")
        if console_output is not None:
            console_output.append(msg)

    return x, history


def newtons_method(func, grad, hessian, x0, tol=1e-5, max_iter=100, console_output=None):
    """
    Newton'un yönteminin uygulanması.

    Parametreler:
        func: Minimize edilecek fonksiyon.
        grad: Fonksiyonun gradyanı (türev fonksiyonu).
        hessian: Hessian matrisi (fonksiyonun ikinci türevleri).
        x0: Başlangıç noktası (numpy array).
        tol: Hata toleransı (küçük değişimler için durma şartı).
        max_iter: Maksimum iterasyon sayısı.
        console_output: Konsol çıktıları için bir liste (arayüze aktarmak için).

    Dönüş:
        x_opt: Optimum çözüm.
        history: Görselleştirme için noktaların listesi.
    """
    x = x0
    history = [x.copy()]

    for i in range(max_iter):
        g = grad(x)
        h = hessian(x)

        if np.linalg.norm(g) < tol:
            msg = f"{i+1}. iterasyonda gradyan normu ile yakınsama sağlandı."
            if console_output is not None:
                console_output.append(msg)
            break

        # Newton adımı
        try:
            delta_x = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            msg = f"Hessian matrisi tekil, çözüm başarısız oldu."
            if console_output is not None:
                console_output.append(msg)
            break

        x_new = x + delta_x
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            msg = f"{i+1}. iterasyonda vektör değişimi ile yakınsama sağlandı."
            if console_output is not None:
                console_output.append(msg)
            break

        x = x_new

        # Konsol çıktısı ile detaylı bilgi
        msg = (f"Iterasyon {i+1}:\n"
               f"    x = {x}\n"
               f"    f(x) = {func(x)}\n"
               f"    Gradyan = {g}\n"
               f"    Adım = {delta_x}")
        if console_output is not None:
            console_output.append(msg)

    return x, history

def create_tabbed_interface():
    """
    Çoklu yöntemler için sekmeli arayüz oluşturur.
    """
    def execute_fletcher_reeves():
        try:
            x0 = np.array([float(x) for x in x0_entry.get().split(",")])
            tol = float(tol_entry.get())
            max_iter = int(max_iter_entry.get())

            console_output.clear()
            _, history = fletcher_reeves_method(function, gradient, x0, tol, max_iter, console_output)
            update_console()
            plot_optimization(function, history)
        except Exception as e:
            console_output.append(f"Hata: {e}")
            update_console()

    def execute_newtons_method():
        try:
            x0 = np.array([float(x) for x in x0_entry2.get().split(",")])
            tol = float(tol_entry2.get())
            max_iter = int(max_iter_entry2.get())

            console_output2.clear()
            _, history = newtons_method(function, gradient, hessian, x0, tol, max_iter, console_output2)
            update_console2()
            plot_optimization(function, history)
        except Exception as e:
            console_output2.append(f"Hata: {e}")
            update_console2()

    def update_console():
        console_textbox.configure(state='normal')
        console_textbox.delete(1.0, tk.END)
        console_textbox.insert(tk.END, "\n".join(console_output))
        console_textbox.configure(state='disabled')

    def update_console2():
        console_textbox2.configure(state='normal')
        console_textbox2.delete(1.0, tk.END)
        console_textbox2.insert(tk.END, "\n".join(console_output2))
        console_textbox2.configure(state='disabled')

    # Tkinter ana pencere
    root = tk.Tk()
    root.title("Optimizasyon Yöntemleri")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    # Tab 1: Fletcher-Reeves
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Fletcher-Reeves")

    tk.Label(tab1, text="Başlangıç Noktası (virgülle ayrılmış):").grid(row=0, column=0, sticky="w")
    x0_entry = tk.Entry(tab1, width=30)
    x0_entry.grid(row=0, column=1)
    x0_entry.insert(0, "0.0, 0.0")

    tk.Label(tab1, text="Tolerans (örn. 1e-5):").grid(row=1, column=0, sticky="w")
    tol_entry = tk.Entry(tab1, width=30)
    tol_entry.grid(row=1, column=1)
    tol_entry.insert(0, "1e-5")

    tk.Label(tab1, text="Maksimum İterasyon:").grid(row=2, column=0, sticky="w")
    max_iter_entry = tk.Entry(tab1, width=30)
    max_iter_entry.grid(row=2, column=1)
    max_iter_entry.insert(0, "100")

    tk.Button(tab1, text="Optimizasyonu Çalıştır", command=execute_fletcher_reeves).grid(row=3, column=0, columnspan=2)

    tk.Label(tab1, text="Konsol Çıktıları:").grid(row=4, column=0, sticky="nw")
    console_textbox = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=20, state='disabled')
    console_textbox.grid(row=4, column=1)

    console_output = []

    # Tab 2: Newton's Method
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Newton's Method")

    tk.Label(tab2, text="Başlangıç Noktası (virgülle ayrılmış):").grid(row=0, column=0, sticky="w")
    x0_entry2 = tk.Entry(tab2, width=30)
    x0_entry2.grid(row=0, column=1)
    x0_entry2.insert(0, "0.0, 0.0")

    tk.Label(tab2, text="Tolerans (örn. 1e-5):").grid(row=1, column=0, sticky="w")
    tol_entry2 = tk.Entry(tab2, width=30)
    tol_entry2.grid(row=1, column=1)
    tol_entry2.insert(0, "1e-5")

    tk.Label(tab2, text="Maksimum İterasyon:").grid(row=2, column=0, sticky="w")
    max_iter_entry2 = tk.Entry(tab2, width=30)
    max_iter_entry2.grid(row=2, column=1)
    max_iter_entry2.insert(0, "100")

    tk.Button(tab2, text="Optimizasyonu Çalıştır", command=execute_newtons_method).grid(row=3, column=0, columnspan=2)

    tk.Label(tab2, text="Konsol Çıktıları:").grid(row=4, column=0, sticky="nw")
    console_textbox2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=80, height=20, state='disabled')
    console_textbox2.grid(row=4, column=1)

    console_output2 = []

    root.mainloop()

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

if __name__ == "__main__":
    create_tabbed_interface()