import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from utils import *
from consts import Consts


# initial
t = np.linspace(0, 15*60*60, 9999)
rtol = 1.49012e-12  # ode precision


# ref traj
oe0 = np.array([408e3 + Consts.rEarth, 0, 0, 0, 0, 0])
n0 = math.sqrt(Consts.muEarth / oe0[0] ** 3)  # mean motion
print(n0)
rv0 = oe2rv(oe0)
traj0 = odeint(central_gravity_motion, rv0, t, rtol=rtol)

orb = 1e3 * np.array([0.0000, 2.5981, 1.5000, -0.0032, 0.0000, 0.0000])
eci = orb_2_eci(rv0, orb, n0)

# plot ref
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(traj0[:, 0], traj0[:, 1], traj0[:, 2])

# hyll tetrahedron traj
K = 1000
a = 0
b = -math.sqrt(5)
c = -K * math.sqrt(10)
A3, B3, C3, D3, E3 = tetrahedron_configuration_1(K, a, b, c)
ht_trajs = list()
for i in range(3):
    traj = hyll_traj(t, n0, A3[i], B3[i], C3[i], D3[i], E3[i])
    print(np.linalg.norm(traj[0:3]))
    ht_trajs.append(traj)

# cg tetrahedron traj
rvs_init = list()
cg_trajs = list()
for i in range(3):
   rv_orb = hyll_traj(np.array([0]), n0, A3[i], B3[i], C3[i], D3[i], E3[i])
   rv_eci = orb_2_eci(rv0, rv_orb[0], n0)
   rvs_init.append(rv_eci)
   traj = odeint(central_gravity_motion, rvs_init[i], t, rtol=rtol)
   cg_trajs.append(traj-traj0)

## quals
ht_qual = np.array([])
for i in range(ht_trajs[0].shape[0]):
    q = tetrahedron_quality(ht_trajs[0][i, 0:3], ht_trajs[1][i, 0:3], ht_trajs[2][i, 0:3])
    ht_qual = np.append(ht_qual, q)

cg_qual = np.array([])
for i in range(cg_trajs[0].shape[0]):
    q = tetrahedron_quality(cg_trajs[0][i, 0:3], cg_trajs[1][i, 0:3], cg_trajs[2][i, 0:3])
    cg_qual = np.append(cg_qual, q)


# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']
for i in range(3):
    traj = cg_trajs[i]
    x = np.array((0, traj[0, 0]))
    y = np.array((0, traj[0, 1]))
    z = np.array((0, traj[0, 2]))
    plt.plot(x, y, z, colors[i])

# for i in range(3):
#     traj = cg_trajs[i]
#     x = np.array((0, traj[-1, 0]))
#     y = np.array((0, traj[-1, 1]))
#     z = np.array((0, traj[-1, 2]))
#     plt.plot(x, y, z, colors[i])

for i in range(3):
    traj = cg_trajs[i]
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]
    plt.plot(x, y, z, colors[i], linewidth=0.3)

# plot qual
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(t/3600, ht_qual)
plt.plot(t/3600, cg_qual)

plt.show()

