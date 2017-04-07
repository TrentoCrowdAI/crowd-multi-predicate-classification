import numpy as np


def do_quiz(trust_min, quiz_papers_n, cheaters_prop):
    easy_add_val = 0.2
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
        worker_accuracy = np.random.uniform(0.5, 1.)

    for paper_id in range(quiz_papers_n):
        # decide if paper in/out of scope
        # bernoulli trial with p=0.5
        gold_value = np.random.binomial(1, 0.5)
        # if a paper is easy, otherwise it is avg
        if worker_type is not 'rand_ch':
            if np.random.binomial(1, 0.5):
                if worker_accuracy + easy_add_val > 1.:
                    worker_accuracy = 1.
                else:
                    worker_accuracy += easy_add_val
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
    statistic_passed = {
        'rand_ch': 0,
        'smart_ch': 0,
        'worker': 0
    }
    statistic_total = {
        'rand_ch': 0,
        'smart_ch': 0,
        'worker': 0
    }
    for _ in range(1000):
        result = do_quiz(trust_min=0.75, quiz_papers_n=4, cheaters_prop=0.3)
        if len(result) > 1:
            statistic_passed[result[2]] += 1
            statistic_total[result[2]] += 1
        else:
            statistic_total[result[0]] += 1

    rand_cheaters_passed = statistic_passed['rand_ch']/float(statistic_total['rand_ch'])*100
    smart_cheaters_passed = statistic_passed['smart_ch']/float(statistic_total['smart_ch'])*100
    workers_passed = statistic_passed['worker']/float(statistic_passed['worker'])*100

    print 'random cheaters passed: {}%'.format(rand_cheaters_passed)
    print 'smart cheaters passed: {}%'.format(smart_cheaters_passed)
    print 'workers passed: {}%'.format(workers_passed)
