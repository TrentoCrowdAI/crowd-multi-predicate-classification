import numpy as np
import pandas as pd

from estimator import Estimator

'''
z - proportion of cheaters
lr - loss ration, i.e., how much a False Negative is more harmful than a False Positive
votes_per_item - crowd votes per item for base round
worker_tests - number of test questions per worker
theta - overall proportion of positive items
filters_num - number of filters
filters_select - selectivity of filters (probability of applying a filter)
filters_dif - difficulty of filters
iter_num - number of iterations for averaging results
---------------------
FP == False Exclusion
FN == False Inclusion
'''


if __name__ == '__main__':
    params = {
        'filters_num': 4,
        'items_num': 1000,
        'baseround_items': 20,
        'items_per_worker': 10,
        'votes_per_item': 3,
        'filters_select': [0.14, 0.14, 0.28, 0.42],
        'filters_dif': [1., 1., 1.1, 0.9],
        'worker_tests': 5,
        'lr': 5,
        'stop_score': 100,
        'iter_num': 50,
        'z': 0.3,
        'theta': 0.3
    }

    estimator = Estimator(params)
    output = estimator.run()
    output.to_csv('output.csv', index=False)
