import numpy as np
import mpmath
import sympy
from scipy.special import kv, kve
import math

def log_besselKv(nu, y):
    return np.log(kve(nu, y)) - np.log(y)
# kve(x, y) would be equal to besselK(x = y, nu = x, expon.scaled = T)

def log_besselKvFA(nu, y):
    val = np.log(kv(nu, y))
    if any(np.isinf(val)):
        val = math.log(sympy.Float(mpmath.besselk(nu, y)))
    return val

def weighted_sum(z, wt):
    return sum(z * wt)

def Rlam(x, lam=None):
    v1 = kve(lam + 1, x) * np.exp(x)
    v0 = kve(lam, x) * np.exp(x)
    val = v1 / v0

    if np.isinf(v1) or np.isinf(v0) or v0 == 0 or v1 == 0:
        lv1 = math.log(sympy.Float(mpmath.besselk(np.abs(lam + 1), x)))
        lv0 = math.log(sympy.Float(mpmath.besselk(np.abs(lam), x)))
        val = np.exp(lv1 - lv0)

    return val

