import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from utils import *
from consts import Consts
from semi_models import *
from plots import *
np.random.seed(200)

# initial
t = np.linspace(0, int(1.55 * 60 * 60), int(1/2 * 1.55 * 60 * 60))
# t = np.linspace(0, 10, 50)
rtol = 1.49012e-12  # ode precision

# ref traj
a0 = 450e3 + Consts.rEarth
ecc0 = 0
incl0 = 87 * np.pi / 180
oe0 = np.array([a0, ecc0, incl0, 0, 0, 0])
n0 = math.sqrt(Consts.muEarth / oe0[0] ** 3)  # mean motion
rv0 = oe2rv(oe0)
traj0 = odeint(central_gravity_motion, rv0, t, rtol=rtol)

# initial tetrahedron construction orb
initial_sat_conf_orb = list()
c1m = 35
c2m = 35
c3m = 35

initial_sat_conf_orb.append([25*c1m,    0*c2m,                   -25 * c3m,                      0])
initial_sat_conf_orb.append([25*c1m,    0*c2m,                    25 * c3m,                      0])
initial_sat_conf_orb.append([75*c1m,    75*np.sqrt(3)/2*c2m,      0 * c3m,                      np.pi/4.0])
initial_sat_conf_orb.append([75*c1m,    75*np.sqrt(3)/2*c2m,      0 * c3m,                      7.0*np.pi/4.0])
support_sat_num = len(initial_sat_conf_orb)


initial_rvs_orb = list()
for i in range(support_sat_num):
    params = initial_sat_conf_orb[i]
    rv = hyll_traj_DA_form_2(t, n0, params[0], params[1], params[2], params[3])
    initial_rvs_orb.append(rv[0])

# central gravity (cg) tetrahedron trajes
cg_trajs_orb = list()
cg_trajs_eci = list()
for i in range(support_sat_num):
    rv_eci = orb_2_eci(rv0, initial_rvs_orb[i], n0)
    traj_eci = odeint(central_gravity_motion, rv_eci, t, rtol=rtol)
    cg_trajs_eci.append(traj_eci)
    dim = np.shape(traj_eci)
    traj_orb = np.empty(dim)
    for k in range(dim[0]):
        rv_orb = eci_2_orb(traj0[k, :], traj_eci[k, :], n0)
        traj_orb[k] = rv_orb

    cg_trajs_orb.append(traj_orb)


# plot_conf_qual(t, cg_trajs_orb, save=True)
# plot_distances_orb(t, cg_trajs_orb, save=True)
# plot_initial_conf_orb(cg_trajs_orb, save=True)
# plot_distances_from_first(t, cg_trajs_orb)
# animate_sat_orb(t, cg_trajs_orb)
# plt.show()


# MF measure
from igrf_utils.igrf_fast import magn_field_ECI, DCM_ECEF_to_ECI
lst_flag = False
lst_depth = 1
lst_dt = 0.2  # s
ex = np.array([1, 0, 0])
b_model_trajs = [np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3])]
b_mes_trajs = [np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3])]
b_idw_trajs = [np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3])]
b_ok_trajs = [np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3]), np.empty([dim[0], 3])]
lat_arg_t = list()
for i in range(len(t)):

    # if i < 1:
    #     continue

    # print(i)

    # e_obj = traj0[i, 0:3]
    # e_obj = e_obj / np.linalg.norm(e_obj)
    # lat_arg = np.arccos(np.dot(ex, e_obj))  # check it!
    lat_arg = n0 * t[i]
    lat_arg_t.append(lat_arg)
    lat_arg = lat_arg % (2 * np.pi)

    # mes
    bs_orb = list()
    bs_noisy_orb = list()
    rs_orb = list()
    for k in range(support_sat_num):

        r_eci = cg_trajs_eci[k][i, 0:3]
        rv_eci = cg_trajs_eci[k][i]
        b_ref_orb = b_igrf_orb(r_eci, incl0, lat_arg)
        bs_orb.append(b_ref_orb)

        noise = np.random.normal(0, 1e2, 3)
        b_ref_noisy = b_ref_orb + noise
        bs_noisy_orb.append(b_ref_noisy)

        rv_orb = eci_2_orb(traj0[i, :], rv_eci, n0)
        rs_orb.append(rv_orb[0:3])

        if lst_flag:
            for lst_i in range(lst_depth):
                time_back = lst_dt * (lst_i+1)
                traj_lst = odeint(central_gravity_motion, rv_eci, [0, -time_back], rtol=rtol)
                rv_lst_eci = traj_lst[-1]
                r_lst_eci = rv_lst_eci[0:3]
                b_lst_ref_orb = b_igrf_orb(r_lst_eci, incl0, lat_arg)
                bs_orb.append(b_lst_ref_orb)

                b_lst_ref_noisy = b_lst_ref_orb + noise
                bs_noisy_orb.append(b_lst_ref_noisy)

                rv_lst_orb = eci_2_orb(traj0[i, :], rv_lst_eci, n0)
                rs_orb.append(rv_lst_orb[0:3])

    if i == 5:
        pass
        # plot_orb_conf_orb(rs_orb, save=True)
        # plt.show()

    # interpolation
    # bs_idw = interpolation_idw(bs_noisy_orb, rs_orb, p=0.1)

    popt = [5.81016054e-14, 3.11716988e-12, 3.93677367e+05, 2.76660624e+00]
    # if lat_arg > 5 * np.pi / 6 or lat_arg <= np.pi / 6:
    #     popt = [5.17790635e-13, 2.41989057e-12, 6.98189856e+05, 2.13108547e+00]
    # elif lat_arg > np.pi / 6 and lat_arg <= np.pi / 2:
    #     popt = [5.35883326e-13, 6.67900796e-12, 5.99176901e+05, 2.52993867e+00]
    # elif lat_arg > np.pi / 2 and lat_arg <= 5 * np.pi / 6:
    #     popt = [4.81266785e-13, 7.27197232e-12, 5.67228982e+05, 2.30457086e+00]

    bs_ok = interpolation_OK(bs_noisy_orb, rs_orb, powered_exponential, popt)

    for k in range(4):
        b_model_trajs[k][i, :] = bs_orb[k]
        b_mes_trajs[k][i, :] = bs_noisy_orb[k]

        # b_idw = bs_idw[k, :]
        # b_idw_trajs[k][i, :] = b_idw

        b_ok = bs_ok[k, :]
        b_ok_trajs[k][i, :] = b_ok

    # # map mes
    # Rs_for_int = np.delete(rs_orb, 1, axis=0)
    # Bs_for_int = np.delete(bs_noisy_orb, 1, axis=0)
    #
    # D = 70
    # step = 1
    # x = np.arange(-D, D, step)
    # y = np.arange(-D, D, step)
    # X, Y = np.meshgrid(x, y)
    #
    # Z = np.zeros(shape=(2*D, 2*D))
    # for ii in range(2*D):
    #     for jj in range(2*D):
    #         r_orb = np.array([X[ii, jj], Y[ii, jj], 0])
    #         ok = ord_kriging(Bs_for_int, Rs_for_int, r_orb, powered_exponential, popt1)
    #         A_eci_2_orb = RM_rci_2_ref(incl0, lat_arg)
    #         r_eci = A_eci_2_orb.T @ (r_orb + np.array([0, 0, a0]))
    #         model = b_igrf_orb(r_eci, incl0, lat_arg)
    #         ok_err = ok - model
    #         Z[ii, jj] = np.linalg.norm(ok_err)
    #
    # im = plt.imshow(Z, cmap=plt.cm.RdBu, extent=(-D, D, D, -D), interpolation='bilinear')
    # plt.colorbar(im)
    # plt.show()

# plot b
# plot_b_mes(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True)
# plot_b_idw(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True)
# plot_b_ok(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True)
# plot_b_ok_vs_mes(lat_arg_t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True, use_lat_arg=True)
# plot_b_idw_minus_ok(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True)

for sat_num in range(4):
    plot_b_ok_vs_mes2(lat_arg_t, b_model_trajs[sat_num][:, 0], b_mes_trajs[sat_num][:, 0], b_ok_trajs[sat_num][:, 0], sat_num + 1, 'x', 'r')
    plot_b_ok_vs_mes2(lat_arg_t, b_model_trajs[sat_num][:, 1], b_mes_trajs[sat_num][:, 1], b_ok_trajs[sat_num][:, 1], sat_num + 1, 'y', 'g')
    plot_b_ok_vs_mes2(lat_arg_t, b_model_trajs[sat_num][:, 2], b_mes_trajs[sat_num][:, 2], b_ok_trajs[sat_num][:, 2], sat_num + 1, 'z', 'b')
    plot_b_ok_vs_mes2_norm(lat_arg_t, b_model_trajs[sat_num], b_mes_trajs[sat_num], b_ok_trajs[sat_num], sat_num + 1, 'abs', 'c')
plt.show()


# ok rms sat 0: 53.27619119347923
# ok rms sat 1: 55.07467829980554
# ok rms sat 2: 54.08665568397436
# ok rms sat 3: 52.952870275649104


