from scipy.special import binom
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random

from scipy.optimize import fmin_tnc
import math
from classifier_utils import find_jt
import pandas as pd
from quiz_simulation import do_quiz_criteria, do_quiz_criteria_confm

cost = 5
J = 30
Nt=5
acc=0.86


# HELPERS
def run_quiz_criteria(quiz_papers_n=4, cheaters_prop=0.3, criteria_num=4):
    acc_passed_distr = []
    for _ in range(100000):
        result = do_quiz_criteria(quiz_papers_n, cheaters_prop, criteria_num)
        if len(result) > 1:
            acc_passed_distr.append(result[0])
    return acc_passed_distr


def run_quiz_criteria_confm(quiz_papers_n, cheaters_prop):
    acc_passed_distr = [[], []]
    for _ in range(100000):
        result = do_quiz_criteria_confm(quiz_papers_n, cheaters_prop)
        if len(result) > 1:
            acc_passed_distr[0].append(result[0])
            acc_passed_distr[1].append(result[1])
    return acc_passed_distr


def find_jt_criteria(theta, J, acc_avg, cost):
    Jt_params = range(J / 2 + 1, J + 1, 1)
    loss_stat = {}
    for Jt in Jt_params:
        loss = estimate_loss_criteria(theta, J, Jt, acc_avg, cost)
        loss_stat.update({loss: Jt})
    Jt_optim = loss_stat[min(loss_stat.keys())]
    # print 'loss_est: {}'.format(min(loss_stat.keys()))
    return Jt_optim, min(loss_stat.keys())


def estimate_loss_criteria(theta, J, Jt, acc_avg, cost):
    p_fe_e = 0.
    criteria_number = 4
    for k in range(Jt, J+1, 1):
        p_fe_e += binom(J, k)*(1-acc_avg)**k*acc_avg**(J-k)
    p_fe_e *= theta
    p_fe = 1 - pow(1-p_fe_e, criteria_number)

    p_fi_e = 0.
    for k in range(J-Jt+1, J+1, 1):
        p_fi_e += binom(J, k)*(1-acc_avg)**k*acc_avg**(J-k)
    p_fi_e *= (1-theta)
    p_fi = 1 - pow(1 - p_fi_e, criteria_number)
    loss = p_fe * cost + p_fi
    return loss


def find_jt_criteria_confm(theta, J, acc_in, acc_out, cost):
    Jt_params = range(J / 2 + 1, J + 1, 1)
    loss_stat = {}
    for Jt in Jt_params:
        loss = estimate_loss_criteria_confm(theta, J, Jt, acc_in, acc_out, cost)
        loss_stat.update({loss: Jt})
    Jt_optim = loss_stat[min(loss_stat.keys())]
    # print 'loss_est: {}'.format(min(loss_stat.keys()))
    return Jt_optim, min(loss_stat.keys())


def estimate_loss_criteria_confm(theta, J, Jt, acc_in, acc_out, cost):
    p_fe_e = 0.
    criteria_number = 4
    for k in range(Jt, J+1, 1):
        p_fe_e += binom(J, k)*(1-acc_in)**k*acc_in**(J-k)
    p_fe_e *= theta
    p_fe = 1 - pow(1-p_fe_e, criteria_number)

    p_fi_e = 0.
    for k in range(J-Jt+1, J+1, 1):
        p_fi_e += binom(J, k)*(1-acc_out)**k*acc_out**(J-k)
    p_fi_e *= (1-theta)
    p_fi = 1 - pow(1 - p_fi_e, criteria_number)
    loss = p_fe * cost + p_fi
    return loss


# FUNCTIONS
def fun(theta, Jt):
    p_fe = 0.
    for k in range(Jt, J+1, 1):
        p_fe += binom(J, k)*(1-acc)**k*acc**(J-k)
    p_fe *= theta
    p_fi = 0.
    for k in range(J-Jt+1, J+1, 1):
        p_fi += binom(J, k)*(1-acc)**k*acc**(J-k)
    p_fi *= (1-theta)
    z = p_fe * cost + p_fi
    return z


def plot_loss():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0.0, 1.1, 0.01)
    y = np.arange(16, 31, 1)
    # x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    Gx, Gy = np.gradient(Z)  # gradients with respect to x and y
    G = (Gx ** 2 + Gy ** 2) ** .5  # gradient magnitude
    N = G / G.max()  # normalize 0..1
    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        facecolors=cm.jet(N),
        linewidth=0, antialiased=False, shade=False)
    # ax.scatter(X, Y, Z)
    # ax.plot_wireframe(X, Y, Z,)


    ax.set_xlabel('Theta')
    ax.set_ylabel('Jt')
    ax.set_zlabel('Loss')
    ax.text2D(0.05, 0.95, "L    oss vs Theta vs Jt", transform=ax.transAxes)

    plt.show()


def loss_vs_tests_scope():
    z = 0.22
    theta = 0.1
    cost = 10
    data = []
    papers_page = 10
    for Nt in range(1, 11, 1):
        for J in [2, 3, 5, 10]:
            Zs = (z * 0.5 ** Nt) / (z * 0.5 ** Nt + (2. * (1 - z) / (Nt + 1)) * (1 - 1. / (2 ** Nt + 1)))
            acc_tw_avg = 2**(Nt+1)*(Nt+1)*(1-0.5**(Nt+2))/((2**(Nt+1)-1)*(Nt+2))
            acc_avg = Zs * 0.5 + (1 - Zs) * acc_tw_avg

            Jt_opt, loss = find_jt(theta, J, acc_avg, cost)
            budget = J*(Nt+papers_page)/float(papers_page)
            data.append([Nt, J, Jt_opt, loss, budget, cost])

    with open('../data/loss_tests_formula_in95.csv', 'a') as f:
        pd.DataFrame(data, columns=['Nt', 'J', 'Jt_opt', 'loss', 'budget', 'cost']).\
            to_csv('../data/loss_tests_formula_11111.csv', index=False, header=False)

#
# def loss_vs_tests_criteria():
#     z = 0.3
#     theta = 0.5
#     cost = 5
#     data = []
#     papers_page = 10
#     criteria_num = 4
#     for Nt in range(1, 11, 1):
#         acc_avg = np.mean(run_quiz_criteria(quiz_papers_n=Nt, cheaters_prop=z, criteria_num=criteria_num))
#         for J in [2, 3, 5, 10]:
#             Jt_opt, loss = find_jt_criteria(theta, J, acc_avg, cost)
#             budget = J * (Nt + papers_page) / float(papers_page)
#             data.append([Nt, J, Jt_opt, loss, budget, cost])
#
#     pd.DataFrame(data, columns=['Nt', 'J', 'Jt_opt', 'loss', 'budget', 'cost']).\
#         to_csv('../data/criteria/loss_tests_criteria.csv', index=False)


def loss_vs_tests_criteria_confm():
    z = 0.3
    theta = 0.5
    cost = 5
    data = []
    papers_page = 10
    for Nt in range(1, 11, 1):
        acc = run_quiz_criteria_confm(quiz_papers_n=Nt, cheaters_prop=z)
        acc_in = np.mean(acc[1])
        acc_out = np.mean(acc[0])
        acc_avg = np.mean([acc_in, acc_out])
        for J in [2, 3, 5, 10]:
            Jt_opt_confm, loss_confm = find_jt_criteria_confm(theta, J, acc_in, acc_out, cost)
            budget = J * (Nt + papers_page) / float(papers_page)
            # data_confm.append([Nt, J, Jt_opt_confm, loss_confm, budget, cost, 'confm'])

            Jt_opt, loss = find_jt_criteria(theta, J, acc_avg, cost)
            data.append([Nt, J, Jt_opt, loss, budget, cost, 'avg'])
            data.append([Nt, J, Jt_opt_confm, loss_confm, budget, cost, 'confm'])

    # pd.DataFrame(data_confm, columns=['Nt', 'J', 'Jt_opt', 'loss', 'budget', 'cost']).\
    #     to_csv('../data/criteria/loss_tests_criteria_confm.csv', index=False)
    pd.DataFrame(data, columns=['Nt', 'J', 'Jt_opt', 'loss', 'budget', 'cost', 'alg']). \
        to_csv('../data/criteria/loss_tests_criteria_theta05.csv', index=False)


if __name__ == '__main__':
    # loss_vs_tests_scope()
    loss_vs_tests_criteria_confm()
    # plot_loss()

