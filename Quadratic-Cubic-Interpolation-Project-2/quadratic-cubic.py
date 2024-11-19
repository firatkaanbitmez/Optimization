import numpy as np

# Küçük bir epsilon değeri tanımlanır, sayısal kararsızlıkları önlemek için
EPSILON = 1e-10

def quadratic_interpolation(f, x0, x1, x2, tolerance=1e-5, max_iter=100):
    """
    Quadratic Interpolation yöntemi ile unimodal bir fonksiyonun minimumunu bulur.

    Parametreler:
    f: Minimumu bulunacak hedef fonksiyon.
    x0, x1, x2: Başlangıç noktaları (farklı olmalıdır ve aralığı kapsamalıdır).
    tolerance: Yakınsama kriteri, minimum x ve f değerleri için fark.
    max_iter: Maksimum iterasyon sayısı.

    Döndürür:
    Minimum nokta tahmini (float).
    """
    print("==== Quadratic Interpolation Başladı ====")
    print(f"Başlangıç noktaları: x0={x0}, x1={x1}, x2={x2}\n")

    for iteration in range(max_iter):
        # Verilen noktalardaki fonksiyon değerlerini hesapla
        f0, f1, f2 = f(x0), f(x1), f(x2)
        print(f"Iterasyon {iteration + 1}: f(x0)={f0:.6f}, f(x1)={f1:.6f}, f(x2)={f2:.6f}")

        # Parabolün katsayılarını hesapla
        try:
            numerator = (f0 * (x1**2 - x2**2) + f1 * (x2**2 - x0**2) + f2 * (x0**2 - x1**2))
            denominator = (f0 * (x1 - x2) + f1 * (x2 - x0) + f2 * (x0 - x1))

            if abs(denominator) < EPSILON:
                raise ValueError("Payda çok küçük, sayısal problemleri önlemek için işlem durduruluyor.")

            # Minimum tahmini x değerini hesapla
            x_min = 0.5 * (numerator / denominator)
            f_min = f(x_min)
            print(f"Tahmini minimum: x_min={x_min:.6f}, f(x_min)={f_min:.6f}")
        except ZeroDivisionError:
            print("Sayısal problem nedeniyle sıfıra bölme hatası oluştu, iterasyon durduruluyor.")
            return None

        # Yakınsama kontrolü
        if abs(x_min - x1) < tolerance and abs(f_min - f1) < tolerance:
            print(f"{iteration + 1}. iterasyonda {x_min:.6f} değerine yakınsandı.\n")
            return x_min

        # Aralığı daraltarak yeni noktaları güncelle
        if x_min < x1:
            if f_min < f1:
                x2, x1 = x1, x_min
            else:
                x0 = x_min
        else:
            if f_min < f1:
                x0, x1 = x1, x_min
            else:
                x2 = x_min

        # Aralık ve noktaların güncellenmesi hakkında detaylı bilgi yazdır
        print(f"Yeni noktalar: x0={x0}, x1={x1}, x2={x2}")
        print(f"Aralık boyutları: |x2 - x0| = {abs(x2 - x0):.6f}\n")

    print("Maksimum iterasyon sayısına ulaşıldı, yakınsama sağlanamadı.\n")
    return x1

def cubic_interpolation(f, x0, x1, x2, x3, tolerance=1e-5, max_iter=100):
    """
    Cubic Interpolation yöntemi ile unimodal bir fonksiyonun minimumunu bulur.

    Parametreler:
    f: Minimumu bulunacak hedef fonksiyon.
    x0, x1, x2, x3: Başlangıç noktaları (farklı ve aralığı kapsayan).
    tolerance: Yakınsama kriteri, minimum x ve f değerleri için fark.
    max_iter: Maksimum iterasyon sayısı.

    Döndürür:
    Minimum nokta tahmini (float).
    """
    print("==== Cubic Interpolation Başladı ====")
    print(f"Başlangıç noktaları: x0={x0}, x1={x1}, x2={x2}, x3={x3}\n")

    for iteration in range(max_iter):
        # Verilen noktalardaki fonksiyon değerlerini hesapla
        f0, f1, f2, f3 = f(x0), f(x1), f(x2), f(x3)
        print(f"Iterasyon {iteration + 1}: f(x0)={f0:.6f}, f(x1)={f1:.6f}, f(x2)={f2:.6f}, f(x3)={f3:.6f}")

        # Katsayı matrisi ve sonucu oluştur
        A = np.array([
            [x0**3, x0**2, x0, 1],
            [x1**3, x1**2, x1, 1],
            [x2**3, x2**2, x2, 1],
            [x3**3, x3**2, x3, 1]
        ])
        b = np.array([f0, f1, f2, f3])

        try:
            # Katsayıları çöz
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Üçüncü dereceden katsayılar çözülürken tekil matrisle karşılaşıldı. Tekrar deneyin veya başlangıç noktalarını değiştirin.")
            return None

        # Polinomun türevini al
        a, b, c, d = coeffs
        derivative = lambda x: 3 * a * x**2 + 2 * b * x + c
        critical_points = np.roots([3 * a, 2 * b, c])

        # Kritik noktaları değerlendir
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
            print("Geçerli bir minimum bulunamadı.\n")
            return None

        print(f"Tahmini minimum: x_min={x_min:.6f}, f(x_min)={min_value:.6f}")

        # Yakınsama kontrolü
        if abs(x_min - x1) < tolerance and abs(min_value - f1) < tolerance:
            print(f"{iteration + 1}. iterasyonda {x_min:.6f} değerine yakınsandı.\n")
            return x_min

        # Aralık güncelle
        if x_min < x1:
            if min_value < f1:
                x3, x2, x1 = x2, x1, x_min
            else:
                x0 = x_min
        else:
            if min_value < f1:
                x0, x1, x2 = x1, x_min, x2
            else:
                x3 = x_min

        # Aralık ve noktaların güncellenmesi hakkında detaylı bilgi yazdır
        print(f"Yeni noktalar: x0={x0}, x1={x1}, x2={x2}, x3={x3}")
        print(f"Aralık boyutları: |x3 - x0| = {abs(x3 - x0):.6f}\n")

    print("Maksimum iterasyon sayısına ulaşıldı, yakınsama sağlanamadı.\n")
    return x1

# Test fonksiyonu ve başlangıç noktaları
if __name__ == "__main__":
    def test_function(x):
        return x**2 - 4*x + 4  # Minimum noktası x=2 olan bir parabol

    print("Quadratic Test:")
    quadratic_interpolation(test_function, 0, 2, 4)

    print("\nCubic Test:")
    def cubic_test_function(x):
        return x**3 - 6*x**2 + 9*x + 1  # Minimum x ~ 1

    cubic_interpolation(cubic_test_function, 0, 1, 2, 3)
