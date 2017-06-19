def input_adapter(responses):
    '''
    :param responses:
    :return: Psi, N
    '''
    Psi = [[] for obj_id in responses.keys()]
    for obj_id, obj_responses in responses.iteritems():
        for worker_id, worker_respons in obj_responses.iteritems():
            Psi[obj_id].append((worker_id, worker_respons[0]))
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
