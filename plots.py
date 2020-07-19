import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.cm as cm
from utils import tetrahedron_quality
from matplotlib import rc
fs = 12
rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})
rc('text', usetex=True)
plt.rc('xtick',labelsize=fs)
plt.rc('ytick',labelsize=fs)

colors = ['r', 'g', 'b', 'c', 'r--', 'g--', 'b--', 'c--']


def plot_initial_conf_orb(cg_trajs_orb, save=True):

    support_sat_num = len(cg_trajs_orb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    ax.set_title('Trajes')

    # cg
    for i in range(support_sat_num):
        traj = cg_trajs_orb[i]
        x = np.array((0, traj[0, 0]))
        y = np.array((0, traj[0, 1]))
        z = np.array((0, traj[0, 2]))
        plt.plot(x, y, z, colors[i])

    for i in range(support_sat_num):
        traj = cg_trajs_orb[i]
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        plt.plot(x, y, z, colors[i], linewidth=0.3)

    if save:
        plt.savefig('pic/sat_conf.png', dpi=800)


def plot_orb_conf_orb(rs_orb, save=True):

    colors = ['ro', 'r*', 'go', 'g*', 'bo', 'b*', 'co', 'c*']
    colors += colors
    colors += colors


    support_sat_num = len(rs_orb)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    ax.set_title('Group')

    # cg
    for i in range(support_sat_num):
        r = rs_orb[i]
        x = [r[0]]
        y = [r[1]]
        z = [r[2]]
        if i < 3:
            plt.plot(x, y, z, 'ro')
        else:
            plt.plot(x, y, z, 'bo')

    if save:
        plt.savefig('pic/mes_conf_with_lst.png', dpi=800)


def plot_conf_qual(t, cg_trajs_orb, save=True):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('t, мин')
    ax.set_ylabel('qual')
    ax.set_title('Качество')

    # quality
    qs = list()
    for i in range(len(t)):
        q = tetrahedron_quality(cg_trajs_orb[0][i, 0:3], cg_trajs_orb[1][i, 0:3], cg_trajs_orb[2][i, 0:3])
        qs.append(q)
    plt.plot(t/60, qs)

    qs = list()
    for i in range(len(t)):
        q = tetrahedron_quality(cg_trajs_orb[0][i, 0:3], cg_trajs_orb[1][i, 0:3], cg_trajs_orb[3][i, 0:3])
        qs.append(q)
    plt.plot(t/60, qs)

    qs = list()
    for i in range(len(t)):
        q = tetrahedron_quality(cg_trajs_orb[0][i, 0:3], cg_trajs_orb[2][i, 0:3], cg_trajs_orb[3][i, 0:3])
        qs.append(q)
    plt.plot(t/60, qs)

    qs = list()
    for i in range(len(t)):
        q = tetrahedron_quality(cg_trajs_orb[1][i, 0:3], cg_trajs_orb[2][i, 0:3], cg_trajs_orb[3][i, 0:3])
        qs.append(q)
    plt.plot(t/60, qs)

    if save:
        plt.savefig('pic/qual.png', dpi=600)


def plot_distances_orb(t, cg_trajs_orb, save=True):

    support_sat_num = len(cg_trajs_orb)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('t, min')
    ax.set_ylabel('r, m')
    ax.set_title('dist')

    # for i in range(support_sat_num):
    #
    #     dist = list()
    #     for k in range(len(t)):
    #         d = np.linalg.norm(cg_trajs_orb[i][k, 0:3])
    #         dist.append(d)

    for i in range(support_sat_num):

        j = i + 1
        if j == support_sat_num:
            j = 0

        dist = list()
        # for k in range(len(t)):
        #     d = np.linalg.norm(cg_trajs_orb[i][k, 0:3] - cg_trajs_orb[j][k, 0:3])
        #     dist.append(d)

        for k in range(len(t)):
            d = np.linalg.norm(cg_trajs_orb[i][k, 0:3])
            dist.append(d)

        plt.plot(t/60, dist, colors[i])

        # lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # if save:
    #     plt.savefig('pic/dists.pdf', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')

def plot_distances_from_first(t, cg_trajs_orb):

    support_sat_num = len(cg_trajs_orb)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('t, мин')
    ax.set_ylabel('r, м')
    # ax.set_ylim([0, 150])
    ax.set_title('Дистанция от первого спутника')

    # for i in range(support_sat_num):
    #
    #     dist = list()
    #     for k in range(len(t)):
    #         d = np.linalg.norm(cg_trajs_orb[i][k, 0:3])
    #         dist.append(d)

    for i in range(support_sat_num-1):

        dist = list()
        for k in range(len(t)):
            d = np.linalg.norm(cg_trajs_orb[0][k, 0:3] - cg_trajs_orb[i+1][k, 0:3])
            dist.append(d)

        plt.plot(t/60, dist, colors[i])


def animate_sat_orb(t, cg_trajs_orb):

        support_sat_num = len(cg_trajs_orb)
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, 30)
        ax.set_xlabel('x, м')
        ax.set_ylabel('y, м')
        ax.set_zlabel('z, м')
        title = ax.set_title('Тетраэдр')

        init_leg_lens = list()
        for i in range(support_sat_num):
            traj = cg_trajs_orb[i]
            l = np.linalg.norm(traj[i, 0])
            init_leg_lens.append(l)

        lim = 1.3 * max(init_leg_lens)
        ax.set_xlim3d([-lim, lim])
        ax.set_ylim3d([-lim, lim])
        ax.set_zlim3d([-lim, lim])

        leg_lines = list()
        for i in range(support_sat_num):
            line, = ax.plot([0, 0, 0], [0, 0, 0], colors[i])
            leg_lines.append(line)

        bottom_lines = list()
        for i in range(support_sat_num):
            line, = ax.plot([0, 0, 0], [0, 0, 0], 'k')
            bottom_lines.append(line)

            def animate(k):
                for i in range(support_sat_num):
                    line = leg_lines[i]
                    traj = cg_trajs_orb[i]
                    line.set_xdata([0, traj[k, 0]])
                    line.set_ydata([0, traj[k, 1]])
                    line.set_3d_properties([0, traj[k, 2]])

                # for i in range(support_sat_num):
                #     line = bottom_lines[i]
                #
                #     j = i + 1
                #     if j == support_sat_num:
                #         j = 0
                #
                #     line.set_xdata([cg_trajs_orb[i][k, 0], cg_trajs_orb[j][k, 0]])
                #     line.set_ydata([cg_trajs_orb[i][k, 1], cg_trajs_orb[j][k, 1]])
                #     line.set_3d_properties([cg_trajs_orb[i][k, 2], cg_trajs_orb[j][k, 2]])

                title.set_text("{:.2f} ч".format(t[k] / 3600))

                return leg_lines + bottom_lines + [title]


        anim = animation.FuncAnimation(fig, animate, interval=2, blit=True, save_count=500)


def plot_b_mes(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True):

    fig = plt.figure()

    for i in range(4):
        ax = fig.add_subplot(221 + i)
        ax.set_title('Измерения ref, cпутник %s' % (i))
        ax.set_ylabel('mu')
        ax.set_xlabel('Время, мин')

        b_idw = b_idw_trajs[i]
        b_model = b_model_trajs[i]
        b_mes = b_mes_trajs[i]
        b_ok = b_ok_trajs[i]

        plt.plot(t / 60, b_mes[:, 0], 'r')
        plt.plot(t / 60, b_mes[:, 1], 'g')
        plt.plot(t / 60, b_mes[:, 2], 'b')

        if save:
            plt.savefig('pic/mes.png', dpi=800)


def plot_b_idw(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True):

    fig = plt.figure()

    for i in range(4):
        ax = fig.add_subplot(221 + i)
        ax.set_title('Ошибка интерполяции IDW, cпутник %s' % (i))
        ax.set_ylabel('dmu')
        ax.set_xlabel('Время, мин')

        b_idw = b_idw_trajs[i]
        b_model = b_model_trajs[i]
        b_mes = b_mes_trajs[i]
        b_ok = b_ok_trajs[i]

        plt.plot(t / 60, b_idw[:, 0] - 1 * b_model[:, 0], 'r')
        plt.plot(t / 60, b_idw[:, 1] - 1 * b_model[:, 1], 'g')
        plt.plot(t / 60, b_idw[:, 2] - 1 * b_model[:, 2], 'b')

        if save:
            plt.savefig('pic/idw.png', dpi=800)


def plot_b_idw_minus_ok(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True):
    fig = plt.figure()

    for i in range(4):
        ax = fig.add_subplot(221 + i)
        ax.set_title('IDW - OK, cпутник %s' % (i))
        ax.set_ylabel('dmu')
        ax.set_xlabel('Время, мин')

        b_idw = b_idw_trajs[i]
        b_model = b_model_trajs[i]
        b_ok = b_ok_trajs[i]

        plt.plot(t / 60, b_idw[:, 0] - 1 * b_ok[:, 0], 'r')
        plt.plot(t / 60, b_idw[:, 1] - 1 * b_ok[:, 1], 'g')
        plt.plot(t / 60, b_idw[:, 2] - 1 * b_ok[:, 2], 'b')

        if save:
            plt.savefig('pic/idw_minus_ok.png', dpi=800)


def plot_b_ok(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True):
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(221 + i)
        ax.set_title('Ошибка интерполяции OK, cпутник %s' % (i))
        ax.set_ylabel('dmu')
        ax.set_xlabel('Время, мин')

        b_idw = b_idw_trajs[i]
        b_model = b_model_trajs[i]
        b_mes = b_mes_trajs[i]
        b_ok = b_ok_trajs[i]

        plt.plot(t / 60, b_ok[:, 0] - 1 * b_model[:, 0], 'r')
        plt.plot(t / 60, b_ok[:, 0] - 1 * b_model[:, 0], 'g')
        plt.plot(t / 60, b_ok[:, 2] - 1 * b_model[:, 2], 'b')

        if save:
            plt.savefig('pic/ok.png', dpi=800)


def plot_b_ok_vs_mes(t, b_model_trajs, b_mes_trajs, b_idw_trajs, b_ok_trajs, save=True, use_lat_arg=False):
    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot(221 + i)
        ax.set_title('Error sat %s' % (i))
        ax.set_ylabel('dmu')

        if use_lat_arg:
            ax.set_xlabel('${\\theta}, ^{\circ}$')
        else:
            ax.set_xlabel('t, min')

        b_idw = b_idw_trajs[i]
        b_model = b_model_trajs[i]
        b_mes = b_mes_trajs[i]
        b_ok = b_ok_trajs[i]

        if use_lat_arg:
            plt.plot(np.array(t) * 180 / np.pi, b_mes[:, 0] - 1 * b_model[:, 0], 'k')
            plt.plot(np.array(t) * 180 / np.pi, b_ok[:, 0] - 1 * b_model[:, 0], 'r')
        else:
            plt.plot(t / 60, b_mes[:, 0] - 1 * b_model[:, 0], 'k')
            plt.plot(t / 60, b_ok[:, 0] - 1 * b_model[:, 0], 'r')

        rms_sat_ok = np.sqrt(np.mean(np.square(b_ok[:, 0] - 1 * b_model[:, 0])))
        rms_sat_mes = np.sqrt(np.mean(np.square(b_mes[:, 0] - 1 * b_model[:, 0])))
        print("ok rms sat %s: %s" % (i, rms_sat_ok))
        # print("mes rms sat %s: %s" % (i, rms_sat_mes))

        # plt.plot(t / 60, b_idw[:, 0] - 1 * b_model[:, 0], 'g')
        # plt.plot(t / 60, b_idw[:, 1] - 1 * b_model[:, 1], 'g')
        # plt.plot(t / 60, b_idw[:, 2] - 1 * b_model[:, 2], 'b')

        if save:
            plt.savefig('pic/ok_vs_mes_2.png', dpi=800)

def plot_b_ok_vs_mes2(t, b_model_vector, b_mes_vector, b_ok_vector, sut_num, y_label, color):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Satellite %s' % (sut_num), fontsize=fs+4)
    ax.set_ylabel('RMSE$_%s$, nT' % (y_label), fontsize=fs)
    ax.set_xlabel('${\\theta}, ^{\circ}$', fontsize=fs+2)

    plt.plot(np.array(t) * 180 / np.pi, b_mes_vector - b_model_vector, 'k', label='mes err, %s' % (y_label))
    plt.plot(np.array(t) * 180 / np.pi, b_ok_vector - b_model_vector, color, label='OK err, %s' % (y_label))
    rms_sat_ok = np.sqrt(np.mean(np.square(b_ok_vector - b_model_vector)))

    lgd = ax.legend(bbox_to_anchor=(0.72, 0.97), loc=2, borderaxespad=0.)
    ax.set_ylim([-500, 500])
    plt.savefig('pic/ok_vs_mes_sat_%s_%s_rms_%.1f.png' % (sut_num, y_label, rms_sat_ok), dpi=450, bbox_extra_artists=(lgd,), bbox_inches='tight')



def plot_b_ok_vs_mes2_norm(t, b_model_vector, b_mes_vector, b_ok_vector, sut_num, y_label, color):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Satellite %s' % (sut_num), fontsize=fs+4)
    ax.set_ylabel('$|$RMSE$|$, nT', fontsize=fs)
    ax.set_xlabel('${\\theta}, ^{\circ}$', fontsize=fs+2)

    errs_mes = list()
    for i in range(len(t)):
        err = np.linalg.norm(b_mes_vector[i, :] - b_model_vector[i, :])
        errs_mes.append(err)

    errs_ok = list()
    for i in range(len(t)):
        err = np.linalg.norm(b_ok_vector[i, :] - b_model_vector[i, :])
        errs_ok.append(err)

    rms_sat_ok = np.sqrt(np.mean(np.square(errs_ok)))

    plt.plot(np.array(t) * 180 / np.pi, errs_mes, 'k', label='mes err, %s' % (y_label))
    plt.plot(np.array(t) * 180 / np.pi, errs_ok, color, label='OK err, %s' % (y_label))
    lgd = ax.legend(bbox_to_anchor=(0.72, 0.97), loc=2, borderaxespad=0.)
    plt.savefig('pic/ok_vs_mes_sat_%s_%s_rms_%.1f.png' % (sut_num, y_label, rms_sat_ok), dpi=450, bbox_extra_artists=(lgd,), bbox_inches='tight')