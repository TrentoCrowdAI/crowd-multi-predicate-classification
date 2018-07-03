import numpy as np
import pandas as pd
import copy

from ShortestMultiRun.helpers.utils import Generator, Workers
from ShortestMultiRun.ShortestMultiRun import ShortestMultiRun


class Estimator:
    def __init__(self, params):
        self.params = params

    def run(self, single_run=False, fixed_votes=False):
        if self.params['baseround_items'] % self.params['items_per_worker']:
            raise ValueError(
                'baseround_items must be a multiple of items_per_worker')
        data = []
        # TODO: do not hardcode this
        self.params['filters_dif'] = [1.0, 1.1, 0.9]
        # S-run algorithm
        loss_smrun_list = []
        cost_smrun_list = []
        worker_tests = self.params['worker_tests']
        votes_per_item = self.params['votes_per_item']
        rec_sm, pre_sm, f_sm, f_sm = [], [], [], []
        nt_ini = 1
        nt_end = 10
        j_values = [3, 5, 10]

        if single_run:
            nt_ini = nt_end = worker_tests

        if fixed_votes or single_run:
            j_values = [votes_per_item]

        for Nt in range(nt_ini, nt_end + 1):
            for J in j_values:
                print('Nt: {}. J: {}'.format(Nt, J))
                params = copy.deepcopy(self.params)
                params['worker_tests'] = Nt
                params['votes_per_item'] = J
                i = 0

                while i != params['iter_num']:
                    print("iteration", i)
                    # quiz, generation votes
                    workers_accuracy = Workers(
                        params['worker_tests'], params['z']).simulate_workers()
                    params.update({'workers_accuracy': workers_accuracy,
                                   'ground_truth': None
                                   })

                    _, ground_truth = Generator(
                        params).generate_votes_gt(params['items_num'])
                    params.update({'ground_truth': ground_truth})

                    try:
                        # s-run
                        loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = ShortestMultiRun(
                            params).run()
                        loss_smrun_list.append(loss_smrun)
                        cost_smrun_list.append(cost_smrun)
                        rec_sm.append(rec_sm_)
                        pre_sm.append(pre_sm_)
                        f_sm.append(f_beta_sm)
                        i += 1
                    except Exception as e:
                        print(e)
                        continue

                data.append([Nt, J, params['lr'], np.mean(loss_smrun_list), np.std(loss_smrun_list),
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
                            columns=['worker_tests', 'votes_per_item', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
                                     'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
                                     'f_beta', 'f_beta_std', 'baseround_items', 'total_items', 'theta', 'filters_num']
                            )
