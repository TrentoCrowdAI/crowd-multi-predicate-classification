import numpy as np
import pandas as pd

from ShortestMultiRun.helpers.utils import Generator, Workers
from ShortestMultiRun.ShortestMultiRun import ShortestMultiRun


class Estimator:
    def __init__(self, params):
        self.params = params

    def run(self):
        if self.params['baseround_items'] % self.params['items_per_worker']:
            raise ValueError(
                'baseround_items must be a multiple of items_per_worker')
        data = []
        # S-run algorithm
        params = self.params
        loss_smrun_list = []
        cost_smrun_list = []
        rec_sm, pre_sm, f_sm, f_sm = [], [], [], []
        for _ in range(params['iter_num']):
            # quiz, generation votes
            workers_accuracy = Workers(
                params['worker_tests'], params['z']).simulate_workers()
            params.update({'workers_accuracy': workers_accuracy,
                           'ground_truth': None
                           })

            _, ground_truth = Generator(
                params).generate_votes_gt(params['items_num'])
            params.update({'ground_truth': ground_truth})

            # s-run
            loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = ShortestMultiRun(
                params).run()
            loss_smrun_list.append(loss_smrun)
            cost_smrun_list.append(cost_smrun)
            rec_sm.append(rec_sm_)
            pre_sm.append(pre_sm_)
            f_sm.append(f_beta_sm)

        data.append([params['worker_tests'], params['worker_tests'], params['lr'], np.mean(loss_smrun_list), np.std(loss_smrun_list),
                     np.mean(cost_smrun_list), np.std(
                         cost_smrun_list), 'Crowd-Ensemble', np.mean(rec_sm),
                     np.std(rec_sm), np.mean(pre_sm), np.std(
                         pre_sm), np.mean(f_sm), np.std(f_sm),
                     params['baseround_items'], params['items_num'], params['theta'], params['filters_num']])

        print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
              'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
              .format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(rec_sm),
                      np.std(rec_sm), np.mean(
                          cost_smrun_list), np.std(cost_smrun_list),
                      np.mean(pre_sm), np.mean(f_sm)))

        return pd.DataFrame(data,
                            columns=['worker_tests', 'worker_tests', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
                                     'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
                                     'f_beta', 'f_beta_std', 'baseround_items', 'total_items', 'theta', 'filters_num']
                            )
