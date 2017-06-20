from generator import generate_responses_gt
from utils import run_quiz_criteria_confm, get_loss_dong
from mrun_algorithm import get_loss_cost_mrun

import pandas as pd
import numpy as np


if __name__ == '__main__':
    z = 0.3
    cost = 5
    n_papers = 1000
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    data = []
    for Nt in range(1, 11, 1):
        print Nt
        for J in [2, 3, 5, 10]:
            # loss_dawid_list = []
            loss_baseline_list = []
            cost_baseline = float(criteria_num) * J * (Nt + n_papers) / n_papers
            loss_mrun_list = []
            cost_mrun_list = []
            for _ in range(2):
                acc = run_quiz_criteria_confm(Nt, z, criteria_difficulty)
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                loss_baseline_list.append(get_loss_dong(responses, criteria_num, n_papers, papers_page, J, GT, cost))
                # get_loss_cost_mrun(responses, criteria_num, n_papers, papers_page, J)
            # print "dawid: {}".format(np.mean(loss_dawid_list))
            print 'dong loss: {} std :{}'.format(np.mean(loss_baseline_list), np.std(loss_baseline_list))
            print 'dong cost: {}'.format(cost)
            print '---------------------'

            # data.append([Nt, J, cost, np.mean(loss_dawid_list), np.std(loss_dawid_list), 'dawid'])
            data.append([Nt, J, cost, np.mean(loss_baseline_list), np.std(loss_baseline_list), cost_baseline, 0., 'baseline'])
    pd.DataFrame(data, columns=['Nt', 'J', 'cost', 'loss_mean', 'loss_std', 'cost_mean', 'cost_std', 'alg']). \
        to_csv('output/data/loss_tests_cr5.csv', index=False)
