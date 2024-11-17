import numpy as np

# Küçük bir epsilon değeri tanımla, sayısal kararsızlıkları önlemek için
EPSILON = 1e-8

def quadratic_interpolation(f, x0, x1, x2, tolerance=1e-5, max_iter=100):
    """
    Tek modlu bir fonksiyonun minimumunu bulmak için Quadratic interpolasyon yöntemi.
    
    Parametreler:
    f: Minimum bulunacak hedef fonksiyon
    x0, x1, x2: Başlangıç noktaları
    tolerance: Yakınsama kriteri
    max_iter: Maksimum iterasyon sayısı
    
    Döndürür:
    Minimum nokta tahmini
    """
    for iteration in range(max_iter):
        # Verilen noktalardaki fonksiyon değerlerini hesapla
        f0, f1, f2 = f(x0), f(x1), f(x2)
        
        # (x0, f0), (x1, f1), (x2, f2) noktalarından geçen bir ikinci dereceden fonksiyon oluştur
        numerator = (f0 * (x1**2 - x2**2) + f1 * (x2**2 - x0**2) + f2 * (x0**2 - x1**2))
        denominator = (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1))
        
        if abs(denominator) < EPSILON:
            raise ValueError("Payda çok küçük, sayısal problemleri önlemek için iterasyon durduruluyor.")
        
        x_min = 0.5 * (numerator / denominator)
        
        # Yakınsama kontrolü
        if abs(x_min - x1) < tolerance and abs(f(x_min) - f1) < tolerance:
            print(f"{iteration + 1}. iterasyonda {x_min} değerine yakınsandı.")
            return x_min
        
        # Yeni tahmine göre noktaları güncelle
        if x_min < x1:
            if f(x_min) < f1:
                x2, x1 = x1, x_min
            else:
                x0 = x_min
        else:
            if f(x_min) < f1:
                x0, x1 = x1, x_min
            else:
                x2 = x_min
    
    print("Maksimum iterasyon sayısına ulaşıldı, yakınsama sağlanamadı.")
    return x1

def cubic_interpolation(f, x0, x1, x2, x3, tolerance=1e-5, max_iter=100):
    """
    Tek modlu bir fonksiyonun minimumunu bulmak için Cubic interpolasyon yöntemi.
    
    Parametreler:
    f: Minimum bulunacak hedef fonksiyon
    x0, x1, x2, x3: Başlangıç noktaları
    tolerance: Yakınsama kriteri
    max_iter: Maksimum iterasyon sayısı
    
    Döndürür:
    Minimum nokta tahmini
    """
    for iteration in range(max_iter):
        # Verilen noktalardaki fonksiyon değerlerini hesapla
        f0, f1, f2, f3 = f(x0), f(x1), f(x2), f(x3)
        
        # (x0, f0), (x1, f1), (x2, f2), (x3, f3) noktalarından geçen bir üçüncü dereceden fonksiyon oluştur
        A = np.array([
            [x0**3, x0**2, x0, 1],
            [x1**3, x1**2, x1, 1],
            [x2**3, x2**2, x2, 1],
            [x3**3, x3**2, x3, 1]
        ])
        b = np.array([f0, f1, f2, f3])
        
        try:
            # Üçüncü dereceden katsayıları çöz
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise ValueError("Üçüncü dereceden katsayılar çözülürken tekil matrisle karşılaşıldı.")
        
        # Üçüncü dereceden polinomun türevi
        a, b, c, d = coeffs
        derivative = lambda x: 3*a*x**2 + 2*b*x + c
        
        # Türev = 0 denklemini çözerek kritik noktaları bul
        critical_points = np.roots([3*a, 2*b, c])
        
        # Aralık içinde bulunan ve en düşük fonksiyon değerini veren gerçek kritik noktayı seç
        x_min = None
        min_value = float('inf')
        for root in critical_points:
            if np.isreal(root) and x0 < root < x3:
                real_root = np.real(root)
                f_root = f(real_root)
                if f_root < min_value:
                    x_min = real_root
                    min_value = f_root
        
        if x_min is None:
            print("Aralık içinde geçerli bir kritik nokta bulunamadı.")
            return None
        
        # Yakınsama kontrolü
        if abs(x_min - x1) < tolerance and abs(f(x_min) - f1) < tolerance:
            print(f"{iteration + 1}. iterasyonda {x_min} değerine yakınsandı.")
            return x_min
        
        # Yeni tahmine göre noktaları güncelle
        if x_min < x1:
            if f(x_min) < f1:
                x3, x2, x1 = x2, x1, x_min
            else:
                x0 = x_min
        else:
            if f(x_min) < f1:
                x0, x1, x2 = x1, x_min, x2
            else:
                x3 = x_min
    
    print("Maksimum iterasyon sayısına ulaşıldı, yakınsama sağlanamadı.")
    return x1

# Örnek kullanım
def example_function(x):
    return x**4 - 14*x**3 + 60*x**2 - 70*x

# Quadratic Interpolation
print("\nQuadratic Interpolasyon:")
x_min_quad = quadratic_interpolation(example_function, 0, 2, 4)
print(f"Tahmini minimum: {x_min_quad}")

# Cubic Interpolation
print("\nCubic Interpolasyon:")
x_min_cubic = cubic_interpolation(example_function, 0, 1, 3, 4)
print(f"Tahmini minimum: {x_min_cubic}")
