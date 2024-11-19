# Project-1: Bracketing Method (Golden Section Search)
import numpy as np

# Sabitler
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # Altın Oran sabiti, minimumu bulmak için kullanılan sabit oran
EPSILON = 1e-8  # Sıfıra bölme hatalarını önlemek için küçük bir değer

# Amaç Fonksiyonu
def f(x):
    # x değeri sıfıra yakınsa, sıfıra bölme hatasından kaçınmak için küçük bir epsilon ekleniyor
    if abs(x) < EPSILON:
        x = np.sign(x) * EPSILON  # x'in işaretine göre küçük bir epsilon ekle
    # Amaç fonksiyonunun değeri hesaplanıp geri döndürülüyor
    return 0.65 - (0.75 / (1 + x**2)) - 0.65 * np.arctan(1 / x)

# Altın Oran Arama Yöntemi (Bracketing Method)
def bracketing_method(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    """
    Altın Oran Arama Yöntemi kullanarak f fonksiyonunun minimumunu bulur.
    
    Parametreler:
    f: Minimum bulunacak amaç fonksiyonu
    a: Aralığın başlangıcı
    b: Aralığın sonu
    tol: Yakınsama toleransı
    max_iter: Maksimum iterasyon sayısı
    verbose: İterasyon detaylarını yazdırmak için
    
    Döndürür:
    Minimum nokta tahmini ve bu noktadaki fonksiyon değeri
    """
    # Hatalı parametre kontrolleri
    if a >= b:
        raise ValueError("Geçersiz aralık: 'a' değeri 'b' değerinden küçük olmalıdır.")
    if tol <= 0:
        raise ValueError("Tolerans pozitif olmalıdır.")
    if max_iter <= 0:
        raise ValueError("Maksimum iterasyon sayısı pozitif olmalıdır.")

    # Başlangıç noktalarının hesaplanması
    # x1 ve x2, aralığın içindeki iki nokta olarak hesaplanır ve altın oran kullanılır
    x1 = b - (b - a) / GOLDEN_RATIO
    x2 = a + (b - a) / GOLDEN_RATIO
    f1, f2 = f(x1), f(x2)  # x1 ve x2'deki fonksiyon değerleri hesaplanır

    # Altın Oran Arama iterasyonları
    iteration = 0
    while abs(b - a) > tol and iteration < max_iter:
        iteration += 1
        # İterasyon bilgileri isteğe bağlı olarak yazdırılır
        if verbose:
            print(f"İterasyon {iteration}: a = {a:.12f}, b = {b:.12f}, x1 = {x1:.12f}, x2 = {x2:.12f}, f1 = {f1:.12f}, f2 = {f2:.12f}")

        # Aralık güncelleme
        # Eğer f1 > f2 ise minimum nokta x2'de veya sağında olmalıdır, bu yüzden a güncellenir
        if f1 > f2:
            a = x1  # a, x1'e güncellenir
            x1 = x2  # x1, x2'ye güncellenir (aralık daralır)
            f1 = f2  # f1, f2'ye güncellenir (hesaplama tekrarı önlenir)
            x2 = a + (b - a) / GOLDEN_RATIO  # Yeni x2 değeri hesaplanır
            f2 = f(x2)  # Yeni x2'de fonksiyon değeri hesaplanır
        else:
            # Eğer f1 <= f2 ise minimum nokta x1'de veya solunda olmalıdır, bu yüzden b güncellenir
            b = x2  # b, x2'ye güncellenir
            x2 = x1  # x2, x1'e güncellenir (aralık daralır)
            f2 = f1  # f2, f1'e güncellenir (hesaplama tekrarı önlenir)
            x1 = b - (b - a) / GOLDEN_RATIO  # Yeni x1 değeri hesaplanır
            f1 = f(x1)  # Yeni x1'de fonksiyon değeri hesaplanır

    # Yakınsama kontrolü
    if verbose:
        if abs(b - a) <= tol:
            print(f"{iteration} iterasyon sonunda yakınsadı.")
        else:
            print("Maksimum iterasyon sayısına ulaşıldı, yakınsama sağlanamadı.")

    # Yakınsama sağlandıktan sonra aralığın ortası minimum nokta olarak döndürülür
    min_point = (a + b) / 2
    min_value = f(min_point)  # Bu noktadaki fonksiyon değeri hesaplanır

    if verbose:
        print(f"Tahmin edilen minimum nokta: x = {min_point:.12f}, f(x) = {min_value:.12f}")
        print("\nSonuç Özeti:")
        print(f"Başlangıç aralığı: [{a:.12f}, {b:.12f}]")
        print(f"Toplam iterasyon sayısı: {iteration}")
        print(f"Son aralık genişliği: {abs(b - a):.12e}")
        print(f"Minimum nokta: x = {min_point:.12f}")
        print(f"Minimum noktadaki fonksiyon değeri: f(x) = {min_value:.12f}")

    return min_point, min_value

# Altın Oran Arama yönteminin örnek kullanımı
def main():
    try:
        # Başlangıç aralığı [a, b]
        a, b = -2.0, 2.0
        # Minimum noktayı bulmak için bracketing_method fonksiyonu çağrılır
        min_point, min_value = bracketing_method(f, a, b, tol=1e-12, max_iter=100, verbose=True)
        # Tahmin edilen minimum nokta ve bu noktadaki fonksiyon değeri yazdırılır
        print(f"\nSonuç: Tahmin edilen minimum nokta: x = {min_point:.12f}, f(x) = {min_value:.12f}")
    except ValueError as e:
        # Hatalı parametre girildiğinde kullanıcıya hata mesajı gösterilir
        print(f"Hata: {e}")

# Ana fonksiyonun çağrılması
if __name__ == "__main__":
    main()
