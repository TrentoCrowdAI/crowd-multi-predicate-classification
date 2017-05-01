import numpy as np


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
