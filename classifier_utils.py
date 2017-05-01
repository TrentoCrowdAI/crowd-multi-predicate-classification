def classifier(psi_obj, psi_w, Jt):
    res = []
    for users_values in psi_obj:
        aggregated_value_mv = max(set(users_values), key=users_values.count)

    return res