# ujur

**A seedless, lightweight synthetic tabular data generator for unique datasets, independent simulation runs, and low-risk prototyping.**

ujur is not a model trained on real sensitive data. It draws entropy directly from CPU nanosecond-level timing jitter — making every run genuinely non-reproducible by default, with snapshot-based reproducibility when needed.

> **No seed by default. Reproducibility by snapshot when needed.**

---

## Where does ujur fit?

| | Faker | SDV / Synthcity | ujur |
|---|---|---|---|
| Learns from real data | ❌ | ✅ | ❌ |
| Privacy-preserving | ❌ | ✅ | ❌ |
| Seed-free | ❌ | ❌ | ✅ |
| Parametric control | ❌ | ❌ | ✅ |
| Zero dependencies | ✅ | ❌ | ✅ |
| NIST SP 800-22 | ❌ | ❌ | ✅ (15/15) |
| Use case | Fake/dummy data | Real-data-like synthetic | Independent, parametric datasets |

**ujur is for:**
- 🎓 Educators who need every student to receive a different dataset
- 🎲 Monte Carlo simulations requiring genuinely independent runs
- 🧪 Researchers building pipelines before accessing real data
- ⚡ Quick prototyping and testing without sensitive data

**ujur is not for:**
- Generating data that resembles a specific real dataset
- Privacy-preserving data release
- Cryptographic key generation

---

## Install

```bash
pip install ujur
```

---

## Quick start

```python
import ujur

# Basic
ujur.randint(1, 6)            # dice roll
ujur.rand(100)                # uniform [0, 1)
ujur.normal(170, 10, 100)     # N(170, 10²)
ujur.choice(['A', 'B', 'C'], 10)

# General synthetic dataset
X = ujur.pro(200, 4, (0, 1))                          # 200 obs, 4 features
X = ujur.pro(200, 3, [(0,100), (0,1), (-10,10)])      # per-feature ranges
X = ujur.pro(200, 3, (0,1), corr=0.7)                 # correlated features
X, y = ujur.pro(200, 3, (0,1), target='regression')   # with target

# Regression with known parameters
# y = 8 + 0.2·X₁ + 3·X₂ + 5·X₃ + noise
X, y = ujur.reg(200, (0,1), params=[8, 0.2, 3, 5])
X, y = ujur.reg(200, (0,1), params=[8, 0.2, 3, 5], noise=0.5)
```

---

## No seed — but reproducible when needed

```python
# Save on generation — returns a unique token
X, y, token = ujur.reg(200, (0,1), params=[8, 0.2, 3, 5], save=True)
# → saves ujur_20260416_143022_961100.json

# Reload the identical dataset later
X, y = ujur.load(token)

# Or save manually
token = ujur.save(X, y)
```

---

## Use case recipes

### 🎓 Econometrics assignment
Each student receives a unique dataset — same structure, different values.
```python
import ujur

# y = 5 + 2·income - 0.3·age + 1.5·education + noise
X, y = ujur.reg(150, [(20000,80000), (22,65), (8,20)],
                params=[5, 0.0002, -0.3, 1.5], noise=0.5)
# Save for grading reference
token = ujur.save(X, y)
```

### 🎲 Monte Carlo simulation
Each run draws from fresh hardware entropy — no seed management needed.
```python
results = []
for _ in range(1000):
    X, y = ujur.reg(100, (0,1), params=[3, 1.5, -2, 0.8])
    # ... your model here
    results.append(score)
```

### 🧪 Pipeline prototyping
Build and test your full analysis pipeline before touching real data.
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = ujur.reg(500, (0,1), params=[10, 2, -1.5, 3, 0.5])
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
```

---

## How it works

ujur collects consecutive nanosecond timestamps from the OS hardware-backed
monotonic clock (`perf_counter_ns`), applies stride sampling (every 10th reading),
and extracts the last 3 digits via `t mod 1000`.

```
t = perf_counter_ns()   # hardware clock — no two runs share the same state
L = t % 1000            # last 3 digits → uniform [0, 999] → entropy source
```

This produces values in 0–999 with near-uniform distribution, passing all
15 NIST SP 800-22 randomness tests with zero external dependencies.

**Why sleep-free consecutive sampling?**
When a sleep interval is introduced between measurements (classical jitter designs),
the distribution collapses to a narrow band with only ~161/1000 unique values.
Sleep-free stride sampling produces all 1,000 possible values uniformly.

---

## No dependencies

ujur uses only `time` and `math` from the Python standard library.
No numpy. No scipy. No installation overhead.

---

## Citation

If you use ujur in your research, please cite:

```
Kandemiş, U. (2026). ujur: A Hardware Jitter-Based Seed-Free Synthetic Data
Generator for Python. Zenodo. https://doi.org/10.5281/zenodo.19746284
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19746284.svg)](https://doi.org/10.5281/zenodo.19746284)

---

## Links

- 📦 PyPI: https://pypi.org/project/ujur
- 💻 GitHub: https://github.com/vibexcode/ujur
- 📄 Preprint: https://doi.org/10.5281/zenodo.19746284

---

## License

MIT
