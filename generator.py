import numpy as np
import random


def synthesize(acc_distribution, n_papers, papers_page, J, theta):
    # generate ground truth
    GT = {}
    for obj in range(n_papers):
        GT[obj] = np.random.binomial(1, theta)

    # generate observations
    Psi = [[] for obj in range(n_papers)]
    pages_n = n_papers / papers_page
    for page_id in range(pages_n):
        for _pointer in range(J):
            worker_id = page_id * papers_page + _pointer
            worker_acc = random.choice(acc_distribution)
            for obj_id in range(page_id*J, page_id*J+J, 1):
                gold_value = GT[obj_id]
                if gold_value:
                    worker_judgment = np.random.binomial(1, worker_acc)
                else:
                    worker_judgment = np.random.binomial(1, 1-worker_acc)
                Psi[obj_id].append((worker_id, worker_judgment))
    return GT, Psi
