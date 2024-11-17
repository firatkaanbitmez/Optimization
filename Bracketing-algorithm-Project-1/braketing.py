import numpy as np

# Altın Oran sabiti
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
EPSILON = 1e-8  # Küçük bir değer, sıfıra bölme hatalarını önlemek için

def f(x):
    if abs(x) < EPSILON:
        x = np.sign(x) * EPSILON  # Sıfıra bölme hatalarını önlemek için küçük bir epsilon ekle
    return 0.65 - (0.75 / (1 + x**2)) - 0.65 * np.arctan(1 / x)

def bracketing_method(f, a, b, n=6, verbose=True):
    """
    Kitapta verilen 5.8 numaralı örneğe uygun olarak bracketing yöntemi ile minimumu bulma.
    
    Parametreler:
    f: Hedef fonksiyon
    a: Aralığın başlangıç noktası
    b: Aralığın bitiş noktası
    n: Toplam iterasyon sayısı
    verbose: İterasyon detaylarını yazdırmak için bayrak
    
    Döndürülen:
    Minimum bulunduğu aralık [x_min, x_max]
    """
    L0 = b - a
    x1 = a + 0.382 * L0
    x2 = b - 0.382 * L0
    f1 = f(x1)
    f2 = f(x2)

    for iteration in range(1, n + 1):
        if verbose:
            print(f"Iterasyon {iteration}: x1 = {x1:.4f}, f(x1) = {f1:.6f}, x2 = {x2:.4f}, f(x2) = {f2:.6f}")

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            L0 = b - a
            x1 = a + 0.382 * L0
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            L0 = b - a
            x2 = b - 0.382 * L0
            f2 = f(x2)

        if verbose:
            print(f"Yeni aralık: [{a:.4f}, {b:.4f}]")

        # Erken durdurma kriteri: Aralık boyutu belirli bir eşik değerden küçükse durdur
        if abs(b - a) < EPSILON:
            break

    if verbose:
        print(f"Minimumun bulunduğu aralık: [{a:.4f}, {b:.4f}]")
    return a, b

a = 0.0
b = 3.0

try:
    x_min_ara = bracketing_method(f, a, b)
    print(f"Minimumun olduğu aralık: {x_min_ara}")
except Exception as e:
    print(str(e))
