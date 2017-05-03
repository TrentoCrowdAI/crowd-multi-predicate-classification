def input_adapter(psi_w, n_papers):
    '''
    :param psi_w: 
    :param n_papers: 
    :return: 
    '''

    Psi = [[] for _ in range(n_papers)]
    for w_id, worker_data in enumerate(psi_w):
        for obj, val in worker_data.iteritems():
            Psi[obj].append((w_id, val))
    return Psi


def invert(N, M, Psi):
    """
    Inverts the observation matrix. Need for performance reasons.
    :param N:
    :param M:
    :param Psi:
    :return:
    """
    inv_Psi = [[] for s in range(N)]
    for obj in range(M):
        for s, val in Psi[obj]:
            inv_Psi[s].append((obj, val))
    return inv_Psi
