import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm
from baseline import baseline
from m_run import m_run
from sm_run import sm_run

if __name__ == '__main__':
    z = 0.3
    cr = 5
    n_papers = 500
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    fr_p_part = 0.1
    data = []
    for Nt in range(1, 11, 1):
        for J in [2, 3, 5, 10]:
    # for Nt in [3]:
    #     for J in [3, 5, 10]:
            print 'Nt: {}. J: {}'.format(Nt, J)
            cost_baseline = (Nt + papers_page * criteria_num) * J / float(papers_page)
            loss_baseline_list = []
            fp_b, tp_b, rec_b, pre_b = [], [], [], []
            loss_mrun_list = []
            cost_mrun_list = []
            fp_m, tp_m, rec_m, pre_m = [], [], [], []
            loss_smrun_list = []
            cost_smrun_list = []
            fp_sm, tp_sm, rec_sm, pre_sm = [], [], [], []
            for _ in range(10):
                # quiz, generation responses
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                # baseline
                loss_baseline, fp_rate_b, tp_rate_b, rec_b_, pre_b_ = baseline(responses, criteria_num, n_papers,
                                                                               papers_page, J, GT, cr)
                loss_baseline_list.append(loss_baseline)
                fp_b.append(fp_rate_b)
                tp_b.append(tp_rate_b)
                rec_b.append(rec_b_)
                pre_b.append(pre_b_)
                # m-run
                loss_mrun, cost_mrun, fp_rate_m, tp_rate_m, rec_m_, \
                pre_m_ = m_run(criteria_num, n_papers, papers_page, J, cr, Nt, acc,
                               criteria_power, criteria_difficulty, GT, fr_p_part)
                loss_mrun_list.append(loss_mrun)
                cost_mrun_list.append(cost_mrun)
                fp_m.append(fp_rate_m)
                tp_m.append(tp_rate_m)
                rec_m.append(rec_m_)
                pre_m.append(pre_m_)
                # sm-run
                loss_smrun, cost_smrun, fp_rate_sm, tp_rate_sm, \
                rec_sm_, pre_sm_ = sm_run(criteria_num, n_papers, papers_page, J, cr, Nt, acc,
                                          criteria_power, criteria_difficulty, GT, fr_p_part)
                loss_smrun_list.append(loss_smrun)
                cost_smrun_list.append(cost_smrun)
                fp_sm.append(fp_rate_sm)
                tp_sm.append(tp_rate_sm)
                rec_sm.append(rec_sm_)
                pre_sm.append(pre_sm_)
            print 'BASELINE  loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}'.\
                format(np.mean(loss_baseline_list), cost_baseline, np.mean(fp_b), np.mean(tp_b),
                       np.mean(rec_b), np.mean(pre_b))

            print 'M-RUN     loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}'.\
                format(np.mean(loss_mrun_list), np.mean(cost_mrun_list), np.mean(fp_m), np.mean(tp_m),
                       np.mean(rec_m), np.mean(pre_m))

            print 'SM-RUN    loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}'.\
                format(np.mean(loss_smrun_list), np.mean(cost_smrun_list), np.mean(fp_sm), np.mean(tp_sm),
                       np.mean(rec_sm), np.mean(pre_sm))
            print '---------------------'

            data.append([Nt, J, cr, np.mean(loss_baseline_list), np.std(loss_baseline_list),
                         np.mean(fp_b), np.mean(tp_b), cost_baseline, 0., 'Baseline',
                         np.mean(rec_b), np.mean(pre_b)])
            data.append([Nt, J, cr, np.mean(loss_mrun_list), np.std(loss_mrun_list), np.mean(fp_m),
                         np.mean(tp_m), np.mean(cost_mrun_list), np.std(cost_mrun_list), 'M-runs',
                         np.mean(rec_m), np.mean(pre_m)])
            data.append([Nt, J, cr, np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(fp_sm),
                         np.mean(tp_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list), 'SM-runs',
                         np.mean(rec_sm), np.mean(pre_sm)])
    pd.DataFrame(data, columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'FPR', 'TPR',
                                'price_mean', 'price_std', 'alg', 'recall', 'precision']). \
                                to_csv('output/data/loss_tests_cr5.csv', index=False)
