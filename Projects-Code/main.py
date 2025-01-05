import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from utils import function, gradient, hessian, plot_optimization
from fletcher_reeves import fletcher_reeves_method
from newtons_method import newtons_method

def create_tabbed_interface():
    def execute_fletcher_reeves():
        try:
            x0 = np.array([float(x) for x in x0_entry.get().split(",")])
            tol = float(tol_entry.get())
            max_iter = int(max_iter_entry.get())

            console_output.clear()
            _, history = fletcher_reeves_method(function, gradient, x0, tol, max_iter, console_output)
            update_console(console_output, console_textbox)
            plot_optimization(function, history)
        except Exception as e:
            console_output.append(f"Hata: {e}")
            update_console(console_output, console_textbox)

    def execute_newtons_method():
        try:
            x0 = np.array([float(x) for x in x0_entry2.get().split(",")])
            tol = float(tol_entry2.get())
            max_iter = int(max_iter_entry2.get())

            console_output2.clear()
            _, history = newtons_method(function, gradient, hessian, x0, tol, max_iter, console_output2)
            update_console(console_output2, console_textbox2)
            plot_optimization(function, history)
        except Exception as e:
            console_output2.append(f"Hata: {e}")
            update_console(console_output2, console_textbox2)

    def update_console(output, textbox):
        textbox.configure(state='normal')
        textbox.delete(1.0, tk.END)
        textbox.insert(tk.END, "\n".join(output))
        textbox.configure(state='disabled')

    root = tk.Tk()
    root.title("Optimizasyon Yöntemleri")
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    # Fletcher-Reeves Sekmesi
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
    tk.Button(tab1, text="Çalıştır", command=execute_fletcher_reeves).grid(row=3, column=0, columnspan=2)
    tk.Label(tab1, text="Konsol Çıktıları:").grid(row=4, column=0, sticky="nw")
    console_textbox = scrolledtext.ScrolledText(tab1, wrap=tk.WORD, width=80, height=20, state='disabled')
    console_textbox.grid(row=4, column=1)
    console_output = []

    # Newton Sekmesi
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Newton")
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
    tk.Button(tab2, text="Çalıştır", command=execute_newtons_method).grid(row=3, column=0, columnspan=2)
    tk.Label(tab2, text="Konsol Çıktıları:").grid(row=4, column=0, sticky="nw")
    console_textbox2 = scrolledtext.ScrolledText(tab2, wrap=tk.WORD, width=80, height=20, state='disabled')
    console_textbox2.grid(row=4, column=1)
    console_output2 = []

    root.mainloop()

if __name__ == "__main__":
    create_tabbed_interface()
