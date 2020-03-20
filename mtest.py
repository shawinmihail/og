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


# initial 86400
t = np.linspace(0, 86400, 10000)
#t = np.linspace(0, 1, 2)

# ref traj
oe0 = np.array([700e3 + Consts.rEarth, 0, get_SSO_inclination(700e3 + Consts.rEarth, 0), 0, 0, 0])
n0 = math.sqrt(Consts.muEarth / oe0[0] ** 3)  # mean motion
rv0_ = oe2rv(oe0)
rv0 = 1e6 * np.array([7.071009000000000, -0.00, -0.00, -0.00, -0.001067888033816, 0.007431735936749])
rtol = 1.49012e-12
traj0 = odeint(central_gravity_motion, rv0, t, rtol=rtol)

orb = 1e3 * np.array([0.0000, 2.598076211353316, 1.500000000000000, -0.003185429937793, 0.0000, 0.0000])
# rv1 = 1e6 * np.array([7.072509000000000, -0.002571662450702, -0.000369529754756, -0.0, -0.001067661498671, 0.007430159414373])
rv1 = orb_2_eci(rv0, orb, n0)
# print(rv1 - rv1_)
traj1 = odeint(central_gravity_motion, rv1, t, rtol=rtol)

dr = traj1[:, 0:3] - traj0[:,0:3]
ndr = list()
for i in range(dr.shape[0]):
    ndr.append(np.linalg.norm(dr[i, :]))

# # plot ref
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(t/3600, ndr)
# plt.plot(t/3600, traj0[:, 0])
# plt.plot(t/3600, traj1[:, 0])
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(traj0[:, 0]-traj1[:, 0], traj0[:, 1]-traj1[:, 1], traj0[:, 2]-traj1[:, 2])
plt.show()