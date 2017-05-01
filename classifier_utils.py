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
