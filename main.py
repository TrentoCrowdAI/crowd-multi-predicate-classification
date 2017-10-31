import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm
# from baseline import baseline
# from m_run import m_run
from sm_run import sm_run

if __name__ == '__main__':
    z = 0.3
    lr = 5
    n_papers = 1000
    papers_page = 10
    # theta = 0.3, theta = 0.8
    criteria_power = {0.3: [0.14, 0.14, 0.28, 0.42],
                      0.8: [0.03, 0.03, 0.06, 0.09]}
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    fr_p_part = 0.05
    data = []
    for Nt in range(1, 11, 1):
        for J in [5]:
            print 'Nt: {}. J: {}'.format(Nt, J)
            for pow_term in [10, -10, 'gt']:
                for theta in [0.3]:
                    cost_baseline = (Nt + papers_page * criteria_num) * J / float(papers_page)
                    loss_baseline_list = []
                    fp_b, tp_b, rec_b, pre_b, f_b = [], [], [], [], []
                    loss_mrun_list = []
                    cost_mrun_list = []
                    fp_m, tp_m, rec_m, pre_m, f_m, f_m = [], [], [], [], [], []
                    loss_smrun_list = []
                    cost_smrun_list = []
                    fp_sm, tp_sm, rec_sm, pre_sm, f_sm, f_sm = [], [], [], [], [], []
                    for _ in range(15):
                        # quiz, generation responses
                        acc = run_quiz_criteria_confm(Nt, z, [1.])
                        responses, GT = generate_responses_gt(n_papers, criteria_power[theta], papers_page,
                                                              J, acc, criteria_difficulty)

                        # # m-run
                        # loss_mrun, cost_mrun, fp_rate_m, tp_rate_m, rec_m_, \
                        # pre_m_, f_beta_m = m_run(criteria_num, n_papers, papers_page, J, lr, Nt, acc,
                        #                          criteria_power, criteria_difficulty, GT, fr_p_part)
                        # loss_mrun_list.append(loss_mrun)
                        # cost_mrun_list.append(cost_mrun)
                        # fp_m.append(fp_rate_m)
                        # tp_m.append(tp_rate_m)
                        # rec_m.append(rec_m_)
                        # pre_m.append(pre_m_)
                        # f_m.append(f_beta_m)

                        # sm-run
                        loss_smrun, cost_smrun, fp_rate_sm, tp_rate_sm, \
                        rec_sm_, pre_sm_, f_beta_sm = sm_run(criteria_num, n_papers, papers_page, J, lr, Nt, acc,
                                                             criteria_power[theta], criteria_difficulty, GT,
                                                             fr_p_part, pow_term)
                        loss_smrun_list.append(loss_smrun)
                        cost_smrun_list.append(cost_smrun)
                        fp_sm.append(fp_rate_sm)
                        tp_sm.append(tp_rate_sm)
                        rec_sm.append(rec_sm_)
                        pre_sm.append(pre_sm_)
                        f_sm.append(f_beta_sm)
                    # print 'M-RUN     loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                    #       'recall: {:1.2f}, precision: {:1.2f}, f_b: {}'.\
                    #     format(np.mean(loss_mrun_list), np.mean(cost_mrun_list), np.mean(fp_m), np.mean(tp_m),
                    #            np.mean(rec_m), np.mean(pre_m), np.mean(f_m))

                    print 'SM-RUN   pow_term: {}, theta {}, loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                          'recall: {:1.2f}, precision: {:1.2f}, f_b: {}'.\
                        format(pow_term, theta, np.mean(loss_smrun_list), np.mean(cost_smrun_list), np.mean(fp_sm), np.mean(tp_sm),
                               np.mean(rec_sm), np.mean(pre_sm), np.mean(f_sm))
                    # data.append([Nt, J, lr, np.mean(loss_mrun_list), np.std(loss_mrun_list), np.mean(fp_m),
                    #              np.mean(tp_m), np.mean(cost_mrun_list), np.std(cost_mrun_list), 'M-runs',
                    #              np.mean(rec_m), np.mean(pre_m), np.mean(f_m)])
                    data.append([Nt, J, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(fp_sm),
                                 np.mean(tp_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list), 'SM-runs',
                                 np.mean(rec_sm), np.mean(pre_sm), np.mean(f_sm), str(pow_term), theta])
                print '---------------------'
    pd.DataFrame(data, columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'FPR', 'TPR',
                                'price_mean', 'price_std', 'alg', 'recall', 'precision', 'f_beta',
                                'pow_term', 'theta']). \
                                to_csv('output/data/power_criteria_experiment.csv', index=False)
