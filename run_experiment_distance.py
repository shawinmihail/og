import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from utils import *
from consts import Consts
from semi_models import *
from plots import *
from experiments.experiment_distance import run_experiment_distance

from igrf_utils.igrf_fast import magn_field_ECI, DCM_ECEF_to_ECI

# initial
t = np.linspace(0, int(1.5 * 60 * 60), int(1/5 * 1.5 * 60 * 60))
# t = np.linspace(0, 1500, 500)
rtol = 1.49012e-12  # ode precision

# ref traj
a0 = 450e3 + Consts.rEarth
ecc0 = 0
incl0 = 87 * np.pi / 180
oe0 = np.array([a0, ecc0, incl0, 0, 0, 0])

n0 = math.sqrt(Consts.muEarth / oe0[0] ** 3)  # mean motion
rv0 = oe2rv(oe0)
traj0 = odeint(central_gravity_motion, rv0, t, rtol=rtol)

# ok params
# popt0 = [5.81016054e-14, 3.11716988e-12, 3.93677367e+05, 2.76660624e+00]
# popt1 = [1.27617535e-13, 3.45337501e-12, 4.42403930e+05, 2.15852834e+00]
# popt2 = [5.35883326e-13, 6.67900796e-12, 5.99176901e+05, 2.52993867e+00]

popt0 = [4.13787066e-14, 2.47789190e-12, 3.63634942e+05, 2.77180058e+00]
popt1 = [1.53932238e-13, 2.86297598e-12, 3.89287543e+05, 2.51784409e+00]
popt2 = [5.01913257e-13, 2.12675006e-12, 3.35439656e+05, 2.72733285e+00]

popts = list()
popts.append(popt0)
popts.append(popt1)
popts.append(popt2)

funcs = list()
funcs.append(powered_exponential)
funcs.append(powered_exponential)
funcs.append(powered_exponential)

# !!!
seed = 200
colors = ['r', 'g', 'b']

sat_dist_k = list()
sat_err_k = list()
for k in range(len(popts)):
    popt = popts[k]
    func = funcs[k]
    sat_dist_s = list()
    sat_err_s = list()

    for s in range(10):
        sat_dist_i = list()
        sat_err_i = list()

        for i in range(15):
            print("k: %s, s: %s, i: %s" % (k, s, i))
            c_mult = 5 + i*10
            sat_dists, sat_errs = run_experiment_distance(oe0, traj0, t, c_mult, rtol, popt, func, seed+s)
            sat_dist_i.append(sat_dists)
            sat_err_i.append(sat_errs)

        sat_dist_s.append(sat_dist_i)
        sat_err_s.append(sat_err_i)

    sat_dist_k.append(sat_dist_s)
    sat_err_k.append(sat_err_s)


# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('l, m')
ax.set_ylabel('rms, nT')

n = 2
for k in range(len(popts)):
    sat_dist_s = sat_dist_k[k]
    sat_err_s = sat_err_k[k]

    rms_avg = [0] * len(sat_dist_i)
    ls_avg = [0] * len(sat_dist_i)
    for s in range(len(sat_dist_s)):
        sat_dist_i = sat_dist_s[s]
        sat_err_i = sat_err_s[s]

        ls = list()
        rms = list()
        for i in range(len(sat_dist_i)):
            ls.append(sat_dist_i[i][n])
            rms.append(sat_err_i[i][n])
            rms_avg[i] += sat_err_i[i][n] / len(sat_dist_s)
            ls_avg[i] += sat_dist_i[i][n] / len(sat_dist_s)
    plt.plot(ls_avg, rms_avg, colors[k], label='popt%s' % (k+1))

ax.legend()
ax.set_ylim([75, 175])
ax.legend()
plt.savefig('pic/rms(l)_100_seeds2.png', dpi=800)
plt.show()
