import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm, get_loss_dong
from m_run import get_loss_cost_mrun
from sm_run import get_loss_cost_smrun

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
            loss_smrun_list = []
            cost_smrun_list = []
            for _ in range(10):
                # quiz, generation responses
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                # baseline
                loss_baseline = get_loss_dong(responses, criteria_num, n_papers, papers_page, J, GT, cr)
                loss_baseline_list.append(loss_baseline)
                # m-run
                loss_mrun, cost_mrun = get_loss_cost_mrun(criteria_num, n_papers, papers_page, J, cr, Nt,
                                                          acc, criteria_power, criteria_difficulty, GT)
                loss_mrun_list.append(loss_mrun)
                cost_mrun_list.append(cost_mrun)
                # sm-run
                loss_smrun, cost_smrun = get_loss_cost_smrun(criteria_num, n_papers, papers_page, J, cr, Nt,
                                                             acc, criteria_power, criteria_difficulty, GT)
                loss_smrun_list.append(loss_smrun)
                cost_smrun_list.append(cost_smrun)
            print 'BASELINE  loss: {} std :{}, price: {}'.format(np.mean(loss_baseline_list),
                                                               np.std(loss_baseline_list), cost_baseline)
            print 'M-RUN  loss: {} std :{}, price: {}, std: {}'.format(np.mean(loss_mrun_list),
                                                                np.std(loss_mrun_list), np.mean(cost_mrun_list),
                                                                np.std(cost_mrun_list))
            print 'SM-RUN  loss: {} std :{}, price: {}, std: {}'.format(np.mean(loss_smrun_list),
                                                                    np.std(loss_smrun_list),
                                                                    np.mean(cost_smrun_list),
                                                                    np.std(cost_smrun_list))
            print '---------------------'

            data.append([Nt, J, cr, np.mean(loss_baseline_list),
                         np.std(loss_baseline_list), cost_baseline, 0., 'Baseline'])
            data.append([Nt, J, cr, np.mean(loss_mrun_list), np.std(loss_mrun_list), np.mean(cost_mrun_list),
                         np.std(cost_mrun_list), 'M-runs'])
            data.append([Nt, J, cr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
                         np.mean(cost_smrun_list), np.std(cost_smrun_list), 'SM-runs'])
    pd.DataFrame(data, columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std', 'alg']). \
        to_csv('output/data/loss_tests_cr5.csv', index=False)
