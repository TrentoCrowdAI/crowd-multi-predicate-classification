import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm, get_loss_dong
from m_run import get_loss_cost_mrun
from sm_run import get_loss_cost_smrun

if __name__ == '__main__':
    z = 0.3
    cr = 5
    n_papers = 500
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
            fi_b = []
            fe_b = []
            loss_mrun_list = []
            cost_mrun_list = []
            fi_m = []
            fe_m = []
            loss_smrun_list = []
            cost_smrun_list = []
            fi_sm = []
            fe_sm = []
            for _ in range(10):
                # quiz, generation responses
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                # baseline
                loss_baseline, fi_rate_b, fe_rate_b = get_loss_dong(responses, criteria_num, n_papers,
                                                                    papers_page, J, GT, cr)
                loss_baseline_list.append(loss_baseline)
                fi_b.append(fi_rate_b)
                fe_b.append(fe_rate_b)
                # m-run
                loss_mrun, cost_mrun, fi_rate_m, fe_rate_m, = get_loss_cost_mrun(criteria_num, n_papers,
                                                             papers_page, J, cr, Nt, acc, criteria_power,
                                                                                criteria_difficulty, GT)
                loss_mrun_list.append(loss_mrun)
                cost_mrun_list.append(cost_mrun)
                fi_m.append(fi_rate_m)
                fe_m.append(fe_rate_m)
                # sm-run
                loss_smrun, cost_smrun, fi_rate_sm, fe_rate_sm = get_loss_cost_smrun(criteria_num, n_papers,
                                                               papers_page, J, cr, Nt, acc, criteria_power,
                                                                                   criteria_difficulty, GT)
                loss_smrun_list.append(loss_smrun)
                cost_smrun_list.append(cost_smrun)
                fi_sm.append(fi_rate_sm)
                fe_sm.append(fe_rate_sm)
            print 'BASELINE  loss: {:1.2f}, price: {:1.2f}, fi_rate: {:1.2f}, fe_rate: {:1.2f}'.\
                format(np.mean(loss_baseline_list), cost_baseline, np.mean(fi_b), np.mean(fe_b))

            print 'M-RUN     loss: {:1.2f}, price: {:1.2f}, fi_rate: {:1.2f}, fe_rate: {:1.2f}'.\
                format(np.mean(loss_mrun_list), np.mean(cost_mrun_list), np.mean(fi_m), np.mean(fe_m))

            print 'SM-RUN    loss: {:1.2f}, price: {:1.2f}, fi_rate: {:1.2f}, fe_rate: {:1.2f}'.\
                format(np.mean(loss_smrun_list), np.mean(cost_smrun_list), np.mean(fi_sm), np.mean(fe_sm))
            print '---------------------'

            data.append([Nt, J, cr, np.mean(loss_baseline_list), np.std(loss_baseline_list),
                         np.mean(fi_b), np.mean(fe_b), cost_baseline, 0., 'Baseline'])
            data.append([Nt, J, cr, np.mean(loss_mrun_list), np.std(loss_mrun_list), np.mean(fi_m),
                         np.mean(fe_m), np.mean(cost_mrun_list), np.std(cost_mrun_list), 'M-runs'])
            data.append([Nt, J, cr, np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(fi_sm),
                         np.mean(fe_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list), 'SM-runs'])
    pd.DataFrame(data, columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'fi_rate', 'fe_rate',
                                'price_mean', 'price_std', 'alg']). \
                                to_csv('output/data/loss_tests_cr5.csv', index=False)
