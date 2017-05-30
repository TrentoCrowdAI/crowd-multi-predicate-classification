import numpy as np


def do_quiz_scope(trust_min, quiz_papers_n, cheaters_prop, easy_add_val):
    correct_judgments = 0
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        # decide if a worker a random cheater or smart one
        rand_cheater_prop = 1.
        if np.random.binomial(1, rand_cheater_prop):
            # random cheater
            worker_type = 'rand_ch'
            worker_accuracy = 0.5
        else:
            # smart cheater
            # TO DO
            worker_type = 'smart_ch'
            worker_accuracy = 0.6
    else:
        worker_type = 'worker'
        worker_accuracy = 0.5 + (np.random.beta(1, 1) * 0.5)
        # worker_accuracy = np.random.uniform(0.5, 0.7)

    for paper_id in range(quiz_papers_n):
        # decide if paper in/out of scope
        # bernoulli trial with p=0.5
        # gold_value = np.random.binomial(1, 0.5)

        worker_accuracy_new = worker_accuracy
        # if a paper is easy, otherwise it is avg
        if worker_type is not 'rand_ch':
            if np.random.binomial(1, 0.5):
                if worker_accuracy + easy_add_val > 1.:
                    worker_accuracy_new = 1.
                else:
                    worker_accuracy_new = worker_accuracy + easy_add_val
        if np.random.binomial(1, worker_accuracy_new):
            correct_judgments += 1
    worker_trust = float(correct_judgments)/quiz_papers_n
    if worker_trust >= trust_min:
        return [worker_trust, worker_accuracy, worker_type]
    return [worker_type]


def do_quiz_criteria(quiz_papers_n, cheaters_prop, criteria_num):
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        worker_type = 'rand_ch'
        worker_accuracy = 0.5
    else:
        worker_type = 'worker'
        worker_accuracy = 0.5 + (np.random.beta(1, 1) * 0.5)

    for paper_id in range(quiz_papers_n):
        for _ in range(criteria_num):
            if not np.random.binomial(1, worker_accuracy):
                return [worker_type]
    return [worker_accuracy, worker_type]


def do_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_num):
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        worker_type = 'rand_ch'
        worker_accuracy_in = 0.5
        worker_accuracy_out = 0.5
    else:
        worker_type = 'worker'
        worker_accuracy_in = 0.5 + (np.random.beta(1, 1) * 0.5)
        worker_accuracy_out = worker_accuracy_in + 0.1 if worker_accuracy_in + 0.1 <= 1. else 1.

    for paper_id in range(quiz_papers_n):
        if paper_id % 2:
            for _ in range(criteria_num):
                if not np.random.binomial(1, worker_accuracy_out):
                    return [worker_type]
        else:
            for _ in range(criteria_num):
                if not np.random.binomial(1, worker_accuracy_out):
                    return [worker_type]
    return [worker_accuracy_out, worker_accuracy_in, worker_type]
