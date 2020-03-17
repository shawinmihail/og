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
t = np.linspace(0, 30*60*60, 9999)
#t = np.linspace(0, 1, 2)

# ref traj
oe0 = np.array([700e3 + Consts.rEarth, 0, get_SSO_inclination(700e3 + Consts.rEarth, 0), 0, 0, 0])
n0 = math.sqrt(Consts.muEarth / oe0[0] ** 3)  # mean motion
rv0 = oe2rv(oe0)
traj0 = odeint(central_gravity_motion, rv0, t)

# plot ref
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.plot(traj0[:, 0], traj0[:, 1], traj0[:, 2])

# hyll tetrahedron traj
A3, B3, C3, D3, E3 = tetrahedron_configuration_1(100, 1, 1, 1)
ht_trajes = list()
rvs3 = list()
for i in range(3):
    traj = hyll_traj(t, n0, A3[i], B3[i], C3[i], D3[i], E3[i])
    ht_trajes.append(traj)

# plot hyll tetrahedron trajes
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for traj in ht_trajes:
#     plt.plot(traj[:, 0], traj[:, 1], traj[:, 2])
# plt.show()

# plot animated
fig_a = plt.figure()
ax_a = p3.Axes3D(fig_a)
a_lines = list()
for i in range(3):
    l, = ax_a.plot([], [], [], lw=3)
    a_lines.append(l)

ax_a.set_xlim3d([-200, 200])
ax_a.set_xlabel('X')
ax_a.set_ylim3d([-200, 200])
ax_a.set_ylabel('Y')
ax_a.set_zlim3d([-200, 200])
ax_a.set_zlabel('Z')

def animate_a(i):

    for k in range(3):
        ht_traj = ht_trajes[k]
        x = np.array([0, ht_traj[i, 0]])
        y = np.array([0, ht_traj[i, 1]])
        z = np.array([0, ht_traj[i, 2]])
        a_lines[k].set_data(x, y)
        a_lines[k].set_3d_properties(z)

    return a_lines

fig_b = plt.figure()
ax_b = p3.Axes3D(fig_b)
line_b, = ax_b.plot([], [], [], lw=3)
scale = 1.5
ax_b.set_xlim3d([-Consts.rEarth * scale, Consts.rEarth * scale])
ax_b.set_xlabel('X')
ax_b.set_ylim3d([-Consts.rEarth * scale, Consts.rEarth * scale])
ax_b.set_ylabel('Y')
ax_b.set_zlim3d([-Consts.rEarth * scale, Consts.rEarth * scale])
ax_b.set_zlabel('Z')

def animate_b(i):
    # x = np.array([0, traj0[i, 0]])
    # y = np.array([0, traj0[i, 1]])
    # z = np.array([0, traj0[i, 2]])
    x = traj0[1:i, 0]
    y = traj0[1:i, 2]
    z = traj0[1:i, 3]

    line_b.set_data(x, y)
    line_b.set_3d_properties(z)
    return line_b,

anim_a = FuncAnimation(fig_a, animate_a, frames=200, interval=20, blit=True)
anim_b = FuncAnimation(fig_b, animate_b, frames=200, interval=100, blit=True)
plt.show()

