# ujur

**A seed-free synthetic data generator for Python.**

Every run produces genuinely different results — no seeds, no reproducibility by accident.  
Randomness is sourced from CPU nanosecond-level timing jitter, not deterministic algorithms.  
Passes all 15 NIST SP 800-22 statistical randomness tests.

---

## Install

```bash
pip install ujur
```

---

## Why ujur?

Most synthetic data tools rely on `numpy.random.seed(42)` — meaning the same data appears every time. This is fine for reproducibility, but problematic when you need:

- Every student to receive a **different dataset** for assignments
- **Monte Carlo simulations** with truly independent runs
- **Non-reproducible experiments** that cannot be reverse-engineered from a seed

ujur solves this by drawing entropy directly from the hardware clock — no seed, no repeat.

---

## Basic usage

```python
import ujur

# Integer
ujur.randint(1, 6)           # single roll
ujur.randint(0, 100, 10)     # list of 10

# Float
ujur.rand(100)               # uniform [0, 1)
ujur.uniform(-5, 5, 50)      # uniform [low, high)

# Normal distribution
ujur.randn(50)               # standard normal
ujur.normal(170, 10, 100)    # mean=170, std=10

# Lists
ujur.choice(['A', 'B', 'C'], 10)
ujur.shuffle([1, 2, 3, 4, 5])
```

---

## Synthetic datasets

### General — `pro()`

```python
# 200 observations, 4 features, all in range (0, 1)
X = ujur.pro(200, 4, (0, 1))

# Each feature in a different range
# Missing ranges inherit the last one
X = ujur.pro(200, 4, [(0, 100), (0, 1), (-10, 10)])

# With correlation (all pairs, r = 0.7)
X = ujur.pro(200, 3, (0, 1), corr=0.7)

# With specific mean and std per feature
X = ujur.pro(200, 3, mean_std=[(50, 5), (170, 7), (70, 10)])

# With target variable — regression
X, y = ujur.pro(200, 3, (0, 1), target='regression')

# With target variable — classification
X, y = ujur.pro(200, 3, (0, 1), target='classification')

# All options combined
X, y = ujur.pro(200, 3,
                ranges=[(0, 100), (0, 1), (-10, 10)],
                mean_std=[(50, 10), (0.5, 0.1), (0, 3)],
                corr=0.7,
                target='regression',
                noise=0.2)
```

### Regression with known parameters — `reg()`

```python
# y = 8 + 0.2*X1 + 3*X2 + 5*X3 + noise,  X in (0, 1)
X, y = ujur.reg(200, (0, 1), params=[8, 0.2, 3, 5])

# First element is the intercept, the rest are coefficients
# Number of features is inferred automatically: p = len(params) - 1

# Different range per feature
X, y = ujur.reg(200, [(0, 100), (0, 1), (-5, 5)],
                params=[10, 0.5, 2, -1], noise=0.5)
```

---

## Save and load (instead of seeds)

```python
# Save while generating — returns a token
X, y, token = ujur.reg(200, (0, 1), params=[8, 0.2, 3, 5], save=True)
# → saves ujur_20260416_143022_961100.json

# Load the same data later
X, y = ujur.load(token)

# Or save manually
token = ujur.save(X, y)
```

---

## How it works

ujur collects consecutive nanosecond timestamps from the OS hardware-backed monotonic clock (`perf_counter_ns`), applies stride sampling (every 10th reading), and extracts the last 3 digits via `t mod 1000`.

This produces values in 0–999 with near-uniform distribution and passes NIST SP 800-22 randomness tests (15/15) with zero external dependencies.

```
t = perf_counter_ns()   # hardware clock
L = t % 1000            # last 3 digits → entropy source
```

---

## No dependencies

ujur uses only the Python standard library — `time` and `math`.  
No numpy, no scipy, no installation overhead.

---

## License

MIT
