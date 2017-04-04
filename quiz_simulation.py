import numpy as np


def do_quiz(trust_min, quiz_papers_n, cheaters_prop):
    correct_judgments = 0
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        # decide if a worker a random cheater or smart one
        rand_cheater_prop = 0.5
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
        worker_accuracy = np.random.uniform(0.0, 1.)

    for paper_id in range(quiz_papers_n):
        # decide if paper in/out of scope
        # bernoulli trial with p=0.5
        gold_value = np.random.binomial(1, 0.5)
        worker_judgment = np.random.binomial(1,
                                             worker_accuracy if gold_value == 1
                                             else 1-worker_accuracy)
        if gold_value == worker_judgment:
            correct_judgments += 1
    worker_trust = float(correct_judgments)/quiz_papers_n
    if worker_trust >= trust_min:
        return (worker_trust, worker_accuracy, worker_type)
    return (worker_type,)


if __name__ == '__main__':
    for _ in range(20):
     print do_quiz(trust_min=0.75, quiz_papers_n=4, cheaters_prop=0.3)
