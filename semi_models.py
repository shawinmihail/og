import numpy as np
from scipy.special import kv as mod_bessel
from scipy.special import gamma

def matern(h, c0, c, alpha, nu):
    with np.errstate(all='ignore'):
        return np.where(h == 0, 0, c0 + c * (
                    1. - 1. / (2 ** (nu - 1.) * gamma(nu)) * (h / alpha) ** nu * mod_bessel(nu, h / alpha)))


def cauchy(h, c0, c, alpha, nu):
    return np.heaviside(h, 0) * (c0 + c * (1. - (1. + (h / alpha) ** 2.) ** (-nu)))


def logistic(h, c0, c, alpha):
    b = (19. * c - c0) / (alpha ** 2. * (c + c0))
    a = b * c

    return np.heaviside(h, 0) * (c0 + a * h ** 2. / (1 + b * h ** 2.))


def power(h, c0, c, nu):
    return np.heaviside(h, 0) * (c0 + c * h ** nu)


def wave(h, c0, c, alpha):
    with np.errstate(all='ignore'):
        return np.where(h == 0, 0, c0 + c * (1 - np.sin(h / alpha) / (h / alpha)))


def hole_effect(h, c0, c, alpha):
    return np.heaviside(h, 0) * (c0 + c * (1. - np.cos(h / alpha)))


def linear(h, c0, c, alpha):
    return np.heaviside(h, 0) * (c0 + np.heaviside(h - alpha, 0.) * (c - c * h / alpha) + c * h / alpha)


def exponential(h, c0, c, alpha):
    return np.heaviside(h, 0) * (c0 + c * (1. - np.exp(-3. * h / alpha)))


def gaussian(h, c0, c, alpha):
    return np.heaviside(h, 0) * (c0 + c * (1. - np.exp(-(3. * h / alpha) ** 2.)))


def powered_exponential(h, c0, c, alpha, nu):
    return np.heaviside(h, 0) * (c0 + c * (1. - np.exp(-(3. * h / alpha) ** nu)))


def circular(h, c0, c, alpha):
    with np.errstate(all='ignore'):
        extr = 2 * c / np.pi * (h / alpha * np.sqrt(1. - (h / alpha) ** 2.) + np.arcsin(h / alpha))
    extr = np.nan_to_num(extr)

    return np.heaviside(h, 0) * (c0 + np.heaviside(h - alpha, 0.) * (c - extr) + extr)


def spherical(h, c0, c, alpha):
    extr = c * (1.5 * h / alpha - 0.5 * (h / alpha) ** 3.)

    return np.heaviside(h, 0) * (c0 + np.heaviside(h - alpha, 0.) * (c - extr) + extr)