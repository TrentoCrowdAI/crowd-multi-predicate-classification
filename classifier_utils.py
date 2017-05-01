import numpy as np
from scipy.special import binom


def classifier(psi_obj, Jt):
    res = []
    for users_values in psi_obj:
        aggregated_value_mv = max(set(users_values), key=users_values.count)
        consent = users_values.count(aggregated_value_mv)
        if consent >= Jt:
            res.append(aggregated_value_mv)
        else:
            res.append(1)
    theta_est = sum(res)/float(len(res))
    return [res, theta_est]


def estimate_accuracy(agg_values, psi_w):
    acc_distr = {}
    for w_id, w_vals in enumerate(psi_w):
        count = 0.
        for obj_id in w_vals.keys():
            if w_vals[obj_id] == agg_values[obj_id]:
                count += 1
        acc_estimation = count/len(w_vals)
        acc_distr.update({w_id: acc_estimation})
    acc_avg = np.mean(acc_distr.values())
    return acc_avg


def estimate_loss(theta, J, Jt, acc_avg, cost):
    p_fe = 0.
    for k in range(Jt, J+1, 1):
        p_fe += binom(J, k)*(1-acc_avg)**k*acc_avg**(J-k)
    p_fe *= theta
    p_fi = 0.
    for k in range(J-Jt+1, J+1, 1):
        p_fi += binom(J, k)*(1-acc_avg)**k*acc_avg**(J-k)
    p_fi *= (1-theta)
    loss = p_fe * cost + p_fi
    return loss


def find_jt(theta, J, acc_avg, cost):
    Jt_params = range(J / 2 + 1, J + 1, 1)
    loss_stat = {}
    for Jt in Jt_params:
        loss = estimate_loss(theta, J, Jt, acc_avg, cost)
        loss_stat.update({loss: Jt})
    Jt_optim = loss_stat[min(loss_stat.keys())]
    print 'loss_est: {}'.format(min(loss_stat.keys()))
    return Jt_optim
