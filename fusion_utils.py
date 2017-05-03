def input_adapter(psi_w, n_papers):
    Psi = [[] for _ in range(n_papers)]
    for w_id, worker_data in enumerate(psi_w):
        for obj, val in worker_data.iteritems():
            Psi[obj].append((w_id, val))
    return Psi
