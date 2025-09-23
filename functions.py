import numpy as np

# axis=-1 -> zrom sumu z posledni dimenze (z)


# Sphere function
def sphere(x: np.ndarray):
    return np.sum(x**2, axis=-1)


# Schwefel function
# https://www.sfu.ca/~ssurjano/schwef.html
def schwefel(x: np.ndarray):
    d = x.shape[-1]  # velikost posledni dimenze (z)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


# Rosenbrock function
# https://www.sfu.ca/~ssurjano/rosen.html
def rosenbrock(x: np.ndarray):
    # Handle both single points and grid inputs
    x1 = x[..., :-1]  # All but last dimension
    x2 = x[..., 1:]  # All but first dimension
    return np.sum(100.0 * (x2 - x1**2) ** 2 + (1 - x1) ** 2, axis=-1)


# Rastrigin function
# https://www.sfu.ca/~ssurjano/rastr.html
def rastrigin(x: np.ndarray):
    d = x.shape[-1]
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


# Griewank function
# https://www.sfu.ca/~ssurjano/griewank.html
def griewank(x: np.ndarray):
    d = x.shape[-1]
    sum_term = np.sum(x**2, axis=-1) / 4000.0
    # arrange = Return evenly spaced values
    denom = np.sqrt(np.arange(1, d + 1, dtype=x.dtype))
    prod_term = np.prod(np.cos(x / denom), axis=-1)
    return sum_term - prod_term + 1.0


# Levy function
# https://www.sfu.ca/~ssurjano/levy.html
def levy(x: np.ndarray):
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    s = w[..., :-1]
    term2 = np.sum((s - 1) ** 2 * (1 + 10 * np.sin(np.pi * s + 1) ** 2), axis=-1)
    term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)
    return term1 + term2 + term3


# Michalewicz function
# https://www.sfu.ca/~ssurjano/michal.html
def michalewicz(x: np.ndarray, m=10):
    d = x.shape[-1]
    i = np.arange(1, d + 1, dtype=x.dtype)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi) ** (2 * m)), axis=-1)


# Zakharov function
# https://www.sfu.ca/~ssurjano/zakharov.html
def zakharov(x: np.ndarray):
    d = x.shape[-1]
    sum1 = np.sum(x**2, axis=-1)
    coeff = 0.5 * np.arange(1, d + 1, dtype=x.dtype)
    sum2 = np.sum(coeff * x, axis=-1)
    return sum1 + sum2**2 + sum2**4


# Ackley function
# https://www.sfu.ca/~ssurjano/ackley.html
def ackley(x: np.ndarray):
    d = x.shape[-1]
    a, b, c = 20.0, 0.2, 2 * np.pi
    sum1 = np.sum(x**2, axis=-1)
    sum2 = np.sum(np.cos(c * x), axis=-1)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
