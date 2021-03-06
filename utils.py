
import math
import numpy as np
from consts import Consts
from igrf_utils.igrf_fast import magn_field_ECI, DCM_ECEF_to_ECI

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


def orb_2_eci(rv0, rv, n):
    # x - along track; y - out-of-plane; z - radial
    z_orb = rv0[0:3] / np.linalg.norm(rv0[0:3])
    y_orb = np.cross(rv0[0:3], rv0[3:6])
    y_orb = y_orb / np.linalg.norm(y_orb)
    x_orb = np.cross(y_orb, z_orb)

    M = np.column_stack((x_orb, y_orb, z_orb))
    de_eci = np.matmul(M, rv[0:3])
    r_eci = rv0[0:3] + np.matmul(M, rv[0:3])
    v_eci = rv0[3:6] + np.matmul(M, (rv[3:6] + np.cross(np.array([0, n, 0]), rv[0:3])))
    rv_eci = np.concatenate((r_eci, v_eci))
    return rv_eci


def eci_2_orb(rv0, rv, n):

    z_orb = rv0[0:3] / np.linalg.norm(rv0[0:3])
    y_orb = np.cross(rv0[0:3], rv0[3:6])
    y_orb = y_orb / np.linalg.norm(y_orb)
    x_orb = np.cross(y_orb, z_orb)

    M = np.column_stack((x_orb, y_orb, z_orb))
    M = np.transpose(M)

    r_orb = np.matmul(M, (rv[0:3] - rv0[0:3]))
    v_orb = np.matmul(M, (rv[3:6] - rv0[3:6]) - np.cross(np.array([0, n, 0]), rv[0:3]))

    rv_orb = np.concatenate((r_orb, v_orb))
    return rv_orb


def hyll_traj(t, n, A, B, C, D, E):
    nu = n * t
    x = 2 * A * np.cos(nu) - 2 * B * np.sin(nu) + C
    y = D * np.sin(nu) + E * np.cos(nu)
    z = A * np.sin(nu) + B * np.cos(nu)

    vx = n * -2 * A * np.sin(nu) - n * 2 * B * np.cos(nu)
    vy = n * D * np.cos(nu) - n * E * np.sin(nu)
    vz = n * A * np.cos(nu) - n * B * np.sin(nu)

    return np.column_stack((x, y, z, vx, vy, vz))


def hyll_traj_DA_form(t, n, C1, C2, alpha):

    nu = n * t
    x = C1 * np.cos(nu + alpha)
    y = C2 * np.sin(nu + alpha)
    z = C1 / 2 * np.sin(nu + alpha)

    vx = - n * C1 * np.sin(nu + alpha)
    vy = n * C2 * np.cos(nu + alpha)
    vz = n * C1 / 2 * np.cos(nu + alpha)

    return np.column_stack((x, y, z, vx, vy, vz))


def hyll_traj_DA_form_2(t, n, C1, C2, C3, alpha):

    nu = n * t
    x = C1 * np.cos(nu + alpha) + C3
    y = C2 * np.sin(nu + alpha)
    z = C1 / 2.0 * np.sin(nu + alpha)

    vx = -n * C1 * np.sin(nu + alpha)
    vy = n * C2 * np.cos(nu + alpha)
    vz = n * C1 / 2.0 * np.cos(nu + alpha)

    return np.column_stack((x, y, z, vx, vy, vz))


def hyll_traj_DA_form_3(t, n, C1, C2, C3, alpha, beta):

    nu = n * t
    x = C1 * np.cos(nu + alpha) + C3
    y = C2 * np.sin(nu + beta)
    z = C1 / 2.0 * np.sin(nu + alpha)

    vx = -n * C1 * np.sin(nu + alpha)
    vy = n * C2 * np.cos(nu + beta)
    vz = n * C1 / 2.0 * np.cos(nu + alpha)

    return np.column_stack((x, y, z, vx, vy, vz))



def tetrahedron_configuration_1(K, a, b, c):
    A3 = K * np.array([1, -0.5, -0.5])
    B3 = K * np.array([0, -math.sqrt(3)/2, math.sqrt(3)/2])
    C3 = c * np.array([1, 1, 1])
    D3 = a * K * np.array([1, -0.5, -0.5]) + b * K * np.array([0, -math.sqrt(3)/2, math.sqrt(3)/2])
    E3 = -b * K * np.array([1, -0.5, -0.5]) + a * K * np.array([0, -math.sqrt(3)/2, math.sqrt(3)/2])
    return A3, B3, C3, D3, E3


def tetrahedron_volume(r1, r2, r3):
    M = np.column_stack((r1,r2, r3))
    V = 1/6 * np.linalg.det(M)
    return V


def tetrahedron_square_length(r1, r2, r3):
    d1 = r2 - r1
    d2 = r3 - r2
    d3 = r1 - r3
    L = np.dot(r1, r1) + np.dot(r2, r2) + np.dot(r3, r3) + np.dot(d1, d1) + np.dot(d2, d2) + np.dot(d3, d3)
    return L


def tetrahedron_quality(r1, r2, r3):
    V = tetrahedron_volume(r1, r2, r3)
    L = tetrahedron_square_length(r1, r2, r3)
    if(V < 0):
        return 0
    Q = 12 * (3 * V) ** (2/3) / L
    return Q


# interpolator utils
def earth_dipole_vector_ref_frame(incl, lat_arg, a, dr):
    k = -np.array([np.cos(lat_arg) * np.sin(incl), np.cos(incl), np.sin(lat_arg) * np.sin(incl)])
    r = np.array([0, 0, a]) + dr
    rn = np.linalg.norm(r)
    MU_e = 7.94e+22
    MU_0 = 1.257e-6
    B = MU_e * MU_0 / (4 * np.pi * rn ** 3) * (3 / (rn ** 2) * np.dot(r, k) * r - k)
    return B


def earth_dipole_vector_own_frame(incl, lat_arg, a, dr):
    B = earth_dipole_vector_ref_frame(incl, lat_arg, a, dr)
    A = A_orb_0_to_j(a, dr)
    Bj = np.matmul(A, B)
    return Bj


def A_orb_0_to_j(a, dr):

    R = np.array([0, 0, a]) + dr
    R_w = np.copy(R)
    R_w[0] = R[2]
    R_w[1] = 0.
    R_w[2] = -R[0]

    r = np.linalg.norm(R)
    r_w = np.linalg.norm(R_w)

    A_x = R_w / r_w
    A_y = np.array([-R[0] * R[1] / r_w, r_w, -R[1] * R[2] / r_w]) / r
    A_z = R / r

    A = np.vstack((A_x, A_y, A_z))

    return A


def idw(Bs, Rs, R_int, p):

    delt = Rs - R_int
    w = np.zeros(len(delt))
    B_int = np.zeros_like(Bs[0])
    coef_sum = 0.

    for (i, delt_i) in enumerate(delt):

        r_i = np.linalg.norm(delt_i)

        if r_i == 0:
            return Bs[i]

        w[i] = r_i ** (-p)
        B_int += Bs[i] * w[i]
        coef_sum += w[i]

    return B_int / coef_sum


def interpolation_idw(Bs, Rs, p):
    Bs_int = np.zeros_like(Rs, dtype=float)

    for (i, R) in enumerate(Rs):
        Rs_for_int = np.delete(Rs, i, axis=0)
        Bs_for_int = np.delete(Bs, i, axis=0)

        Bs_int[i] = idw(Bs_for_int, Rs_for_int, R, p)

    return Bs_int


def ord_kriging(Bs, Rs, R_int, func, popt):
    # semivar_matr = np.zeros_like(Rs[0], dtype=float)
    semivar_matr = np.zeros((1, len(Rs)), dtype=float)

    for R in Rs:
        col = np.linalg.norm(Rs - R, axis=1)
        semivar_matr = np.vstack((semivar_matr, col))

    semivar_matr = func(semivar_matr[1:].T, *popt)
    semivar_vec = np.linalg.norm(Rs - R_int, axis=1)
    semivar_vec = func(semivar_vec, *popt)

    matrix = np.ones((Rs.shape[0] + 1, Rs.shape[0] + 1))
    matrix[1:, :-1] = semivar_matr
    matrix[0, -1] = 0

    vector = np.hstack((1, semivar_vec))

    coefs = (np.linalg.inv(matrix) @ vector)[:-1]

    sum = 0

    for (k, B) in zip(coefs, Bs):
        sum += k * B

    return sum


def interpolation_OK(Bs, Rs, func, popt):

    Bs_int = np.zeros_like(Rs, dtype=float)
    for (i, R) in enumerate(Rs):
        Rs_for_int = np.delete(Rs, i, axis=0)
        Bs_for_int = np.delete(Bs, i, axis=0)

        Bs_int[i] = ord_kriging(Bs_for_int, Rs_for_int, R, func, popt)

    return Bs_int


def ecef_psi_theta_r(ecef_xyz):
    phi = np.arctan2(ecef_xyz[1], ecef_xyz[0])
    theta = np.arctan2(np.sqrt(ecef_xyz[0] ** 2 + ecef_xyz[1] ** 2), ecef_xyz[2])
    r = np.sqrt(ecef_xyz[0] ** 2 + ecef_xyz[1] ** 2 + ecef_xyz[2] ** 2)
    return phi, theta, r


def RM_rci_2_ref(inc, latarg):
    A1 = np.array([-np.sin(latarg), np.cos(latarg)*np.cos(inc), np.cos(latarg)*np.sin(inc)])
    A2 = np.array([0, -np.sin(inc), np.cos(inc)])
    A3 = np.array([np.cos(latarg), np.sin(latarg) * np.cos(inc), np.sin(latarg) * np.sin(inc)])
    A = np.vstack((A1, A2, A3))
    return A


def b_igrf_orb(r_eci, incl, lat_arg):
    JD = 2458964.016262
    A_ecef_2_eci = DCM_ECEF_to_ECI(JD)
    A_eci_2_orb = RM_rci_2_ref(incl, lat_arg)
    r_ecef = A_ecef_2_eci.T @ r_eci
    phi, theta, r = ecef_psi_theta_r(r_ecef)
    b_eci = magn_field_ECI(JD, r, theta, phi)
    b_ref = A_eci_2_orb @ b_eci
    return b_ref


