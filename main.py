import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from utils import *
from consts import Consts


# initial
t = np.linspace(0, 1*60*60, 1 * 1000)
rtol = 1.49012e-12  # ode precision


# ref traj
a0 = 408e3 + Consts.rEarth
ecc0 = 0
incl0 = get_SSO_inclination(a0, ecc0)
oe0 = np.array([a0, ecc0, incl0, 0, 0, 0])
n0 = math.sqrt(Consts.muEarth / oe0[0] ** 3)  # mean motion

rv0 = oe2rv(oe0)
traj0 = odeint(central_gravity_motion, rv0, t, rtol=rtol)


# hyll tetrahedron traj
K = 1000
a = 0
b = -math.sqrt(5)
c = -K * math.sqrt(10)
A3, B3, C3, D3, E3 = tetrahedron_configuration_1(K, a, b, c)

ht_trajs = list()
leg_lengths = list()
for i in range(3):
    traj = hyll_traj(t, n0, A3[i], B3[i], C3[i], D3[i], E3[i])
    l = np.linalg.norm(traj[0, 0:3])
    leg_lengths.append(l)
    ht_trajs.append(traj)


# central gravity (cg) tetrahedron traj
rvs_init = list()
cg_trajs = list()
for i in range(3):
    rv_orb = hyll_traj(np.array([0]), n0, A3[i], B3[i], C3[i], D3[i], E3[i])
    rv_eci = orb_2_eci(rv0, rv_orb[0], n0)
    rvs_init.append(rv_eci)
    traj_eci = odeint(central_gravity_motion, rvs_init[i], t, rtol=rtol)

    dim = np.shape(traj_eci)
    traj_orb = np.empty(dim)
    for k in range(dim[0]):
        rv_orb = eci_2_orb(traj0[k, :], traj_eci[k, :], n0)
        traj_orb[k] = rv_orb

    cg_trajs.append(traj_orb)


# traj quals
ht_qual = np.array([])
for i in range(ht_trajs[0].shape[0]):
    q = tetrahedron_quality(ht_trajs[0][i, 0:3], ht_trajs[1][i, 0:3], ht_trajs[2][i, 0:3])
    ht_qual = np.append(ht_qual, q)

cg_qual = np.array([])
for i in range(cg_trajs[0].shape[0]):
    q = tetrahedron_quality(cg_trajs[0][i, 0:3], cg_trajs[1][i, 0:3], cg_trajs[2][i, 0:3])
    cg_qual = np.append(cg_qual, q)


# magnetic field
dim = np.shape(traj0)
ex = np.array([1, 0, 0])
for i in range(dim[0]):
    e_obj = traj0[i, 0:3]
    e_obj = e_obj / np.linalg.norm(e_obj)
    lat_arg = np.arccos(np.dot(ex, e_obj))

    bs = list()
    As = list()
    rs = list()
    for k in range(3):
        traj = cg_trajs[k]
        dr = traj[i, 0:3]

        b_own = earth_dipole_vector_own_frame(incl0, lat_arg, a0, dr)
        b_ref = earth_dipole_vector_ref_frame(incl0, lat_arg, a0, dr)
        A = A_orb_0_to_j(a0, dr)

        bs.append(b_ref)
        rs.append(dr)
        As.append(A)

    m = 2
    r_for_idw = rs[m] + 0*np.array([100, 0, 0])
    b0_int = idw(bs, rs, r_for_idw, p=0.1)
    # print(rs)
    print('---')
    print(bs[m])
    print(b0_int)
    print(b0_int - bs[m])


# # plot ----------------------------------------------------------------------------------
# # trajes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x, м')
# ax.set_ylabel('y, м')
# ax.set_zlabel('z, м')
# ax.set_title('Траектории')
# colors = ['r', 'g', 'b']
#
# # cg
# for i in range(3):
#     traj = cg_trajs[i]
#     x = np.array((0, traj[0, 0]))
#     y = np.array((0, traj[0, 1]))
#     z = np.array((0, traj[0, 2]))
#     plt.plot(x, y, z, colors[i])
#
# for i in range(3):
#     traj = cg_trajs[i]
#     x = traj[:, 0]
#     y = traj[:, 1]
#     z = traj[:, 2]
#     plt.plot(x, y, z, colors[i], linewidth=0.3)
#
# # ht
# for i in range(3):
#     traj = ht_trajs[i]
#     x = traj[:, 0]
#     y = traj[:, 1]
#     z = traj[:, 2]
#     plt.plot(x, y, z, 'k', linewidth=0.3)
#
# # for i in range(3):
# #     traj = ht_trajs[i]
# #     x = np.array((0, traj[0, 0]))
# #     y = np.array((0, traj[0, 1]))
# #     z = np.array((0, traj[0, 2]))
# #     plt.plot(x, y, z, colors[i])
#
# # plot qual
# fig = plt.figure()
# ax = fig.add_subplot()
# plt.plot(t/3600, ht_qual)
# plt.plot(t/3600, cg_qual)
# ax.set_xlabel('t, ч')
# ax.set_ylabel('q')
# ax.set_title('Качество')
#
# # animate tet
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(30, 30)
# ax.set_xlabel('x, м')
# ax.set_ylabel('y, м')
# ax.set_zlabel('z, м')
# title = ax.set_title('Тетраэдр')
#
# lim = 2000
# ax.set_xlim3d([-lim, lim])
# ax.set_ylim3d([-lim, lim])
# ax.set_zlim3d([-lim, lim])
#
# leg_lines = list()
# for i in range(3):
#     line, = ax.plot([0, 0, 0], [0, 0, 0], colors[i])
#     leg_lines.append(line)
#
# bottom_lines = list()
# for i in range(3):
#     line, = ax.plot([0, 0, 0], [0, 0, 0], 'k')
#     bottom_lines.append(line)
#
# def animate(k):
#     for i in range(3):
#         line = leg_lines[i]
#         traj = cg_trajs[i]
#         line.set_xdata([0,  traj[k, 0]])
#         line.set_ydata([0,  traj[k, 1]])
#         line.set_3d_properties([0,  traj[k, 2]])
#
#     for i in range(3):
#         line = bottom_lines[i]
#
#         j = i + 1
#         if j == 3:
#             j = 0
#
#         line.set_xdata([cg_trajs[i][k, 0], cg_trajs[j][k, 0]])
#         line.set_ydata([cg_trajs[i][k, 1], cg_trajs[j][k, 1]])
#         line.set_3d_properties([cg_trajs[i][k, 2], cg_trajs[j][k, 2]])
#
#     title.set_text("{:.2f} ч".format(t[k]/3600))
#
#     return leg_lines + bottom_lines + [title]
#
# anim = animation.FuncAnimation(fig, animate, interval=2, blit=True, save_count=500)
# #anim.save('og.html', fps=15)
#
# plt.show()

