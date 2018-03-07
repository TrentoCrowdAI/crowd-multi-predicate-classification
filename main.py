import numpy as np
import pandas as pd

from ShortestMultiRun.helpers.utils import Generator, Workers
from ShortestMultiRun.ShortestMultiRun import ShortestMultiRun

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
    z = 0.3
    items_num = 1000
    items_per_worker = 10
    baseround_items = 20  # must be a multiple of items_per_worker
    if baseround_items % items_per_worker:
        raise ValueError('baseround_items must be a multiple of items_per_worker')
    worker_tests = 5
    votes_per_item = 3
    lr = 5
    filters_num = 4
    theta = 0.3
    filters_select = [0.14, 0.14, 0.28, 0.42]
    filters_dif = [1., 1., 1.1, 0.9]
    iter_num = 50
    data = []

    params = {
        'filters_num': filters_num,
        'items_num': items_num,
        'baseround_items': baseround_items,
        'items_per_worker': items_per_worker,
        'votes_per_item': votes_per_item,
        'filters_select': filters_select,
        'filters_dif': filters_dif,
        'worker_tests': worker_tests,
        'lr': lr,
        'stop_score': 100
    }

    # S-run algorithm
    loss_smrun_list = []
    cost_smrun_list = []
    rec_sm, pre_sm, f_sm, f_sm = [], [], [], []
    for _ in range(iter_num):
        # quiz, generation votes
        workers_accuracy = Workers(worker_tests, z).simulate_workers()
        params.update({'workers_accuracy': workers_accuracy,
                       'ground_truth': None
                       })

        _, ground_truth = Generator(params).generate_votes_gt(items_num)
        params.update({'ground_truth': ground_truth})

        # s-run
        loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = ShortestMultiRun(params).run()
        loss_smrun_list.append(loss_smrun)
        cost_smrun_list.append(cost_smrun)
        rec_sm.append(rec_sm_)
        pre_sm.append(pre_sm_)
        f_sm.append(f_beta_sm)

    data.append([worker_tests, worker_tests, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
                 np.mean(cost_smrun_list), np.std(cost_smrun_list), 'Crowd-Ensemble', np.mean(rec_sm),
                 np.std(rec_sm), np.mean(pre_sm), np.std(pre_sm), np.mean(f_sm), np.std(f_sm),
                 baseround_items, items_num, theta, filters_num])

    print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
          'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
          .format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(rec_sm),
                  np.std(rec_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list),
                  np.mean(pre_sm), np.mean(f_sm)))

    pd.DataFrame(data,
                 columns=['worker_tests', 'worker_tests', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
                          'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
                          'f_beta', 'f_beta_std', 'baseround_items', 'total_items', 'theta', 'filters_num']
                 ).to_csv('output.csv', index=False)
