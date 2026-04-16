"""
ujur — Jitter-based Random Number Generator
============================================
Donanım bağımsız, seed'siz, tekrarlanamaz rastgele sayı üretici.
Kaynak: CPU nanosaniye sayacının stride örneklemesi (t mod 1000).

Kullanım:
    import ujur

    # Temel
    ujur.randint(1, 6)
    ujur.rand(100)
    ujur.randn(50)
    ujur.normal(170, 10, 100)

    # Veri üretimi
    X = ujur.pro(200, 4, (0, 1))
    X = ujur.pro(200, 4, [(0,1),(10,100),(-5,5),(0,50)])

    # mean/std + korelasyon + hedef
    X, y = ujur.pro(200, 3,
                    ranges=[(0,100),(0,1),(-10,10)],
                    mean_std=[(50,10),(0.5,0.1),(0,3)],
                    corr=[[1,0.7,0],[0.7,1,0],[0,0,1]],
                    target='regression',
                    noise=0.1)
"""

import time
import math

# ── İç çekirdek ──────────────────────────────────────────────────────────────

def _collect(n, stride=10):
    needed = n * stride
    raw = [time.perf_counter_ns() for _ in range(needed)]
    return [raw[i] % 1000 for i in range(0, needed, stride)]

def _to_float(L_values):
    return [v / 1000.0 for v in L_values]

def _box_muller(n):
    result = []
    while len(result) < n:
        L = _collect(2)
        u1 = max(L[0] / 1000.0, 1e-10)
        u2 = L[1] / 1000.0
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        result.append(z0)
        if len(result) < n:
            result.append(z1)
    return result[:n]

def _cholesky(matrix):
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(matrix[i][i] - s, 1e-12))
            else:
                L[i][j] = (matrix[i][j] - s) / (L[j][j] + 1e-12)
    return L

def _apply_correlation(Z, L_chol):
    p = len(Z[0])
    return [[sum(L_chol[i][j] * row[j] for j in range(i+1)) for i in range(p)] for row in Z]

def _scale(z, mean, std, low, high, use_range, use_ms):
    if use_ms:
        val = mean + std * z
    else:
        val = z
    if use_range:
        if use_ms:
            val = max(low, min(high, val))
        else:
            t = max(0.0, min(1.0, (z + 3.0) / 6.0))
            val = low + t * (high - low)
    return val

def _resolve_ranges(p, ranges):
    """Aralıkları p uzunluğuna tamamla — eksik olanlar son aralığı alır."""
    if ranges is None:
        return None
    if isinstance(ranges, tuple) and not isinstance(ranges[0], tuple):
        return [ranges] * p
    rng = list(ranges)
    if len(rng) < p:
        rng += [rng[-1]] * (p - len(rng))
    return rng[:p]

# ── Public API ────────────────────────────────────────────────────────────────

def randint(low, high, size=1):
    """
    [low, high] arasında tam sayı üret.

    Örnek
    -----
    ujur.randint(1, 6)
    ujur.randint(0, 100, 10)
    """
    if low > high:
        raise ValueError("low > high olamaz.")
    span = high - low + 1
    result = [low + (v % span) for v in _collect(size)]
    return result[0] if size == 1 else result

def rand(size=1):
    """
    [0.0, 1.0) arasında uniform float üret.

    Örnek
    -----
    ujur.rand()
    ujur.rand(100)
    """
    result = _to_float(_collect(size))
    return result[0] if size == 1 else result

def randn(size=1):
    """
    Standart normal dağılım (mean=0, std=1).

    Örnek
    -----
    ujur.randn()
    ujur.randn(50)
    """
    result = _box_muller(size)
    return result[0] if size == 1 else result

def uniform(low=0.0, high=1.0, size=1):
    """
    [low, high) arasında uniform float üret.

    Örnek
    -----
    ujur.uniform(-1, 1, 20)
    """
    floats = _to_float(_collect(size))
    result = [low + f * (high - low) for f in floats]
    return result[0] if size == 1 else result

def normal(mean=0.0, std=1.0, size=1):
    """
    Belirtilen mean ve std ile normal dağılım.

    Örnek
    -----
    ujur.normal(170, 10, 100)
    """
    result = [mean + std * z for z in _box_muller(size)]
    return result[0] if size == 1 else result

def choice(population, size=1, replace=True):
    """
    Bir listeden rastgele eleman seç.

    Örnek
    -----
    ujur.choice(['A','B','C'], 10)
    ujur.choice(range(100), 5, replace=False)
    """
    pop = list(population)
    n = len(pop)
    if not replace and size > n:
        raise ValueError("replace=False iken size > len(population) olamaz.")
    if replace:
        result = [pop[randint(0, n-1)] for _ in range(size)]
    else:
        pool = pop[:]
        for i in range(size):
            j = randint(i, n-1)
            pool[i], pool[j] = pool[j], pool[i]
        result = pool[:size]
    return result[0] if size == 1 else result

def shuffle(lst):
    """
    Listeyi yerinde karıştır (in-place).

    Örnek
    -----
    data = [1,2,3,4,5]
    ujur.shuffle(data)
    """
    n = len(lst)
    for i in range(n-1, 0, -1):
        j = randint(0, i)
        lst[i], lst[j] = lst[j], lst[i]

def pro(n, p, ranges=None, *, mean_std=None, corr=None,
        target=None, noise=0.1, save=False):
    """
    n gözlem, p feature veri üret.

    Parametreler
    ------------
    n        : int         — gözlem sayısı
    p        : int         — feature sayısı
    ranges   : tuple veya list[tuple]
               (0,1)                    → tüm featurelar 0-1
               [(0,1),(10,100),(-5,5)]  → her feature farklı aralık
               Eksik featurelar son aralığı miras alır
    mean_std : list[tuple] — her feature için (mean, std)
               Eksik olanlar (0,1) alır
    corr     : list[list]  — p x p korelasyon matrisi
    target   : str veya None
               None           → sadece X döner
               'regression'   → (X, y) döner, y sürekli
               'classification' → (X, y) döner, y 0/1
    noise    : float — hedef gürültüsü (sadece target!=None ise)

    Döndürür
    --------
    X            : list[list[float]]  — (n x p)
    (X, y) tuple : target belirtilmişse

    Örnek
    -----
    # Sadece X
    X = ujur.pro(200, 4, (0, 1))
    X = ujur.pro(200, 4, [(0,1),(10,100),(-5,5)])

    # mean/std ile
    X = ujur.pro(200, 3, mean_std=[(50,5),(170,7),(70,10)])

    # Korelasyonlu
    X = ujur.pro(200, 3, corr=[[1,0.8,0.2],[0.8,1,0.1],[0.2,0.1,1]])

    # Hedef ile (regresyon)
    X, y = ujur.pro(200, 3, ranges=(0,1), target='regression', noise=0.1)

    # Hepsi birden
    X, y = ujur.pro(200, 3,
                    ranges=[(0,100),(0,1),(-10,10)],
                    mean_std=[(50,10),(0.5,0.1),(0,3)],
                    corr=[[1,0.7,0],[0.7,1,0],[0,0,1]],
                    target='regression',
                    noise=0.2)
    """
    rng = _resolve_ranges(p, ranges)
    use_range = rng is not None
    use_ms = mean_std is not None

    if use_ms:
        ms = list(mean_std)
        if len(ms) < p:
            ms += [(0.0, 1.0)] * (p - len(ms))
    else:
        ms = [(0.0, 1.0)] * p

    # Bağımsız standart normal matris (n x p)
    cols = [_box_muller(n) for _ in range(p)]
    Z = [[cols[j][i] for j in range(p)] for i in range(n)]

    # Korelasyon uygula
    if corr is not None:
        if isinstance(corr, (int, float)):
            corr_matrix = [[1.0 if i == j else float(corr) for j in range(p)] for i in range(p)]
        else:
            corr_matrix = corr
        L_chol = _cholesky(corr_matrix)
        Z = _apply_correlation(Z, L_chol)

    # Her feature'ı dönüştür
    X = []
    for row in Z:
        new_row = []
        for i in range(p):
            mean_i, std_i = ms[i]
            low_i  = rng[i][0] if use_range else None
            high_i = rng[i][1] if use_range else None
            val = _scale(row[i], mean_i, std_i, low_i, high_i, use_range, use_ms)
            new_row.append(round(val, 6))
        X.append(new_row)

    if target is None:
        if save:
            token = _save_data(X)
            return X, token
        return X

    # Hedef üret
    coeffs = [uniform(-2, 2) for _ in range(p)]
    intercept = uniform(-1, 1)

    if target == 'regression':
        y = []
        for row in X:
            val = intercept + sum(c * x for c, x in zip(coeffs, row))
            val += normal(0, noise * 3)
            y.append(round(val, 4))

    elif target == 'classification':
        scores = [intercept + sum(c*x for c,x in zip(coeffs,row)) for row in X]
        mn, mx = min(scores), max(scores)
        y = [1 if 1/(1+math.exp(-((s-mn)/(mx-mn+1e-10)*6-3))) > 0.5 else 0
             for s in scores]
    else:
        raise ValueError("target 'regression', 'classification' veya None olmalı.")

    if save:
        token = _save_data(X, y)
        return X, y, token
    return X, y

def reg(n, ranges=None, *, params, noise=0.1, save=False):
    """
    Belirtilen regresyon parametrelerine göre sentetik veri üret.

    y = params[0] + params[1]*X1 + params[2]*X2 + ... + e

    Parametreler
    ------------
    n      : int          — gözlem sayısı
    ranges : tuple/list   — feature aralığı (opsiyonel)
             (0,1)                   → tüm featurelar 0-1
             [(0,1),(10,100),(-5,5)] → her feature farklı aralık
    params : list         — [intercept, b1, b2, ..., bp]
             İlk eleman intercept, geri kalanlar katsayılar
             Feature sayısı otomatik: p = len(params) - 1
    noise  : float        — hata terimi std (varsayılan 0.1)

    Döndürür
    --------
    X : list[list[float]]  — (n x p) feature matrisi
    y : list[float]        — hedef değişken

    Örnek
    -----
    # y = 8 + 0.2*X1 + 3*X2 + 5*X3 + e,  X ∈ (0,1)
    X, y = ujur.reg(200, (0,1), params=[8, 0.2, 3, 5])

    # Farklı aralıklar + gürültü
    X, y = ujur.reg(200, [(0,100),(0,1),(-5,5)],
                    params=[10, 0.5, 2, -1], noise=0.5)

    # Aralık belirtmeden (standart normal)
    X, y = ujur.reg(200, params=[3, 1.5, -2], noise=0.2)
    """
    intercept = params[0]
    coeffs    = params[1:]
    p         = len(coeffs)

    if p == 0:
        raise ValueError("params en az 2 eleman içermeli: [intercept, b1, ...]")

    rng = _resolve_ranges(p, ranges) if ranges is not None else None
    use_range = rng is not None

    # Bağımsız standart normal matris (n x p)
    cols = [_box_muller(n) for _ in range(p)]
    Z = [[cols[j][i] for j in range(p)] for i in range(n)]

    # Feature'ları aralığa dönüştür
    X = []
    for row in Z:
        new_row = []
        for i in range(p):
            if use_range:
                low_i, high_i = rng[i]
                t = max(0.0, min(1.0, (row[i] + 3.0) / 6.0))
                val = low_i + t * (high_i - low_i)
            else:
                val = row[i]
            new_row.append(round(val, 6))
        X.append(new_row)

    # y = intercept + b1*X1 + b2*X2 + ... + e
    y = []
    for row in X:
        val = intercept + sum(b * x for b, x in zip(coeffs, row))
        val += normal(0, noise)
        y.append(round(val, 4))

    if save:
        token = _save_data(X, y)
        return X, y, token
    return X, y


def info():
    """Kütüphane hakkında bilgi yazdır."""
    print("ujur v1.0 — Jitter-based RNG")
    print("Kaynak : CPU nanosaniye sayacı (perf_counter_ns)")
    print("Yöntem : Stride örnekleme (her 10. timestamp) + t mod 1000")
    print("Seed   : YOK — her çalıştırmada farklı sonuç")
    print("Fonksiyonlar:")
    print("  Temel : randint, rand, randn, uniform, normal, choice, shuffle")
    print("  Veri  : pro(n, p, ranges, mean_std, corr, target, noise)")


def _save_data(X, y=None):
    import json
    ts  = time.strftime("%Y%m%d_%H%M%S")
    uid = ''.join(str(v) for v in _collect(4))[:6]
    token = f"ujur_{ts}_{uid}"
    data  = {"token": token, "X": X, "y": y}
    with open(token + ".json", "w") as f:
        json.dump(data, f)
    print(f"Kaydedildi → {token}.json")
    return token


def save(X, y=None):
    """
    Üretilen veriyi dosyaya kaydet, token döndür.

    Döndürür
    --------
    token : str — kayıt dosyasının adı (yüklemek için kullan)

    Örnek
    -----
    token = ujur.save(X, y)
    token = ujur.save(X)
    """
    return _save_data(X, y)


def load(token):
    """
    Token ile kaydedilmiş veriyi geri yükle.

    Örnek
    -----
    X, y = ujur.load(token)
    X, y = ujur.load("ujur_20260416_143022_a3f7")
    """
    import json
    with open(token + ".json", "r") as f:
        data = json.load(f)
    print(f"Yüklendi ← {token}.json")
    return data["X"], data["y"]
