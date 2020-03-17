
import math
import numpy as np
from consts import Consts

def get_SSO_inclination(a, ecc):
    # The formula for the RAAN drift is taken from D.A. Vallado Fundamentals of
    # Astrodynamics, page 649, eq 9-37

    # Inputs: constants, a - semi-major axis [m] and inclination
    # Outputs: SSO orbit inclination [rad]

    p = a * (1 - ecc ** 2)
    n = math.sqrt(Consts.muEarth / a ** 3)

    inclination = math.acos(-(Consts.EarthMeanMotion * 2 * p ** 2) / (3 * n * Consts.rEarth ** 2 * Consts.J2))
    return inclination


def mean2ecc(M, e):
    # Converts mean anomaly to eccentric anomaly.
    #
    # Inputs:
    #   * mean anomaly
    #   * eccentricity
    # Output:
    #   * eccentric anomaly

    E = 0
    # initial guess
    if ((M > -math.pi) and (M < 0)) or (M > math.pi):
        E = M - e
    else:
        E = M + e

    # iteration
    tol = 1e-12
    d = -(E - e*math.sin(E) - M)/(1 - e*math.cos(E))
    while abs(d) >= tol:
        E = E + d
        d = -(E - e*math.sin(E) - M)/(1 - e*math.cos(E))
    return E


def oe2rv(oe):
    # Converts orbital elements (units: radians) to ECI state
    #
    # Input:
    #   * orbital elements
    #   * gravitational parameter
    #   * time math.since given mean anomaly [days]
    # Output:
    #   * ECI state [km, km/s], column vector

    sma = oe[0]  # km
    ecc = oe[1]  # -
    inc = oe[2]  # rad
    RAAN = oe[3]  # rad
    AOP = oe[4]  # rad
    MA = oe[5]  # rad

    E = mean2ecc(MA, ecc)
    v = 2 * math.atan(math.sqrt((1 + ecc) / (1 - ecc)) * math.tan(E / 2))
    r = sma * (1 - ecc ** 2) / (1 + ecc * math.cos(v))

    r_pqw = r * np.array([math.cos(v), math.sin(v), 0])
    v_pqw = math.sqrt(Consts.muEarth / (sma * (1 - ecc ** 2))) * np.array([-math.sin(v), ecc + math.cos(v), 0])

    Rz_O = np.array([[math.cos(RAAN), -math.sin(RAAN), 0], [math.sin(RAAN), math.cos(RAAN), 0], [0, 0, 1]])
    Rx_i = np.array([[1, 0, 0], [0, math.cos(inc), -math.sin(inc)], [0, math.sin(inc), math.cos(inc)]])
    Rz_w = [[math.cos(AOP), -math.sin(AOP), 0], [math.sin(AOP), math.cos(AOP), 0], [0, 0, 1]]
    R = np.matmul(np.matmul(Rz_O, Rx_i), Rz_w)

    r_ijk = np.transpose(np.matmul(R, np.transpose(r_pqw)))
    v_ijk = np.transpose(np.matmul(R, v_pqw))

    sv = np.concatenate((np.transpose(r_ijk), (np.transpose(v_ijk))))
    return sv


def central_gravity_motion(rv, t):

    rv_prime = np.array([rv[3], rv[4], rv[5],
                -Consts.muEarth * rv[0] / (rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2) ** (3 / 2),
                -Consts.muEarth * rv[1] / (rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2) ** (3 / 2),
                -Consts.muEarth * rv[2] / (rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2) ** (3 / 2)])

    return rv_prime


def hyll_traj(t, n, A, B, C, D, E):
    nu = n * t
    x = 2 * A * np.cos(nu) - 2 * B * np.sin(nu) + C
    y = D * np.sin(nu) + E * np.cos(nu)
    z = A * np.sin(nu) + B * np.cos(nu)
    return np.transpose(np.array([x, y, z]))


def tetrahedron_configuration_1(K, a, b, c):
    A3 = K * np.array([1, -0.5, -0.5])
    B3 = K * np.array([0, -math.sqrt(3)/2, math.sqrt(3)/2])
    C3 = c * np.array([1, 1, 1])
    D3 = a * K * np.array([1, -0.5, -0.5]) + b * K * np.array([0, -math.sqrt(3)/2, math.sqrt(3)/2])
    E3 = -b * K * np.array([1, -0.5, -0.5]) + a * K * np.array([0, -math.sqrt(3)/2, math.sqrt(3)/2])
    return A3, B3, C3, D3, E3



