import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm, get_loss_dong
from mrun_algorithm import get_loss_cost_mrun

if __name__ == '__main__':
    z = 0.3
    cr = 5
    n_papers = 1000
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    data = []
    for Nt in range(1, 11, 1):
        for J in [2, 3, 5, 10]:
            print 'Nt: {}. J: {}'.format(Nt, J)
            loss_baseline_list = []
            cost_baseline = (Nt + papers_page * criteria_num) * J / float(papers_page)
            loss_mrun_list = []
            cost_mrun_list = []
            for _ in range(10):
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                loss_baseline = get_loss_dong(responses, criteria_num, n_papers, papers_page, J, GT, cr)
                loss_baseline_list.append(loss_baseline)
                loss_mrun, cost_mrun = get_loss_cost_mrun(criteria_num, n_papers, papers_page, J, cr, Nt,
                                                          acc, criteria_power, criteria_difficulty, GT)
                loss_mrun_list.append(loss_mrun)
                cost_mrun_list.append(cost_mrun)
            print 'BASELINE loss: {} std :{}, cost: {}'.format(np.mean(loss_baseline_list),
                                                               np.std(loss_baseline_list), cost_baseline)
            print 'M-RUN loss: loss: {} std :{}, cost: {}, std: {}'.format(np.mean(loss_mrun_list),
                                                                    np.std(loss_mrun_list), np.mean(cost_mrun_list),
                                                                    np.std(cost_mrun_list))
            print '---------------------'

            data.append([Nt, J, cr, np.mean(loss_baseline_list),
                         np.std(loss_baseline_list), cost_baseline, 0., 'Baseline'])
            data.append([Nt, J, cr, np.mean(loss_mrun_list),
                         np.std(loss_mrun_list), np.mean(cost_mrun_list), np.std(cost_mrun_list), 'M-runs'])
    pd.DataFrame(data, columns=['Nt', 'J', 'cost', 'loss_mean', 'loss_std', 'cost_mean', 'cost_std', 'alg']). \
        to_csv('output/data/loss_tests_cr5.csv', index=False)
