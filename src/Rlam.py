import numpy as np
import mpmath
import sympy
from scipy.special import kve


def Rlam(x, lam=None):
    v1 = kve(lam + 1, x) * np.exp(x)
    v0 = kve(lam, x) * np.exp(x)
    val = v1 / v0

    if np.isinf(v1) or np.isinf(v0) or v0 == 0 or v1 == 0:
        lv1 = np.log(sympy.Float(mpmath.besselk(np.abs(lam + 1), x)))
        lv0 = np.log(sympy.Float(mpmath.besselk(np.abs(lam), x)))
        val = np.exp(lv1 - lv0)

    return val