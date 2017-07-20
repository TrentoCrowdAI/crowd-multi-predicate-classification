import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm
from baseline import baseline_dong, baseline_dawid, baseline_mv

if __name__ == '__main__':
    z = 0.3
    cr = 5
    n_papers = 1000
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

            loss_baseline_list_dawid = []
            fp_d, tp_d, rec_d, pre_d = [], [], [], []

            loss_baseline_list_mv = []
            fp_mv, tp_mv, rec_mv, pre_mv = [], [], [], []
            for _ in range(10):
                # quiz, generation responses
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                # baseline dong
                loss_baseline, fp_rate_b, tp_rate_b, rec_b_, pre_b_ = baseline_dong(responses, criteria_num, n_papers,
                                                                               papers_page, J, GT, cr)
                loss_baseline_list.append(loss_baseline)
                fp_b.append(fp_rate_b)
                tp_b.append(tp_rate_b)
                rec_b.append(rec_b_)
                pre_b.append(pre_b_)

                # baseline dawid
                loss_baseline_d, fp_rate_d, tp_rate_d, rec_d_, pre_d_ = baseline_dawid(responses,
                                                                      criteria_num, n_papers, GT, cr)
                loss_baseline_list_dawid.append(loss_baseline_d)
                fp_d.append(fp_rate_d)
                tp_d.append(tp_rate_d)
                rec_d.append(rec_d_)
                pre_d.append(pre_d_)

                # baseline mv
                loss_baseline_mv, fp_rate_mv, tp_rate_mv, rec_mv_, pre_mv_ = baseline_mv(responses, criteria_num, n_papers,
                                                                                         papers_page, J, GT, cr)
                loss_baseline_list_mv.append(loss_baseline_mv)
                fp_mv.append(fp_rate_mv)
                tp_mv.append(tp_rate_mv)
                rec_mv.append(rec_mv_)
                pre_mv.append(pre_mv_)
            print 'BASELINE DONG  loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}'.\
                format(np.mean(loss_baseline_list), cost_baseline, np.mean(fp_b), np.mean(tp_b),
                       np.mean(rec_b), np.mean(pre_b))
            print 'BASELINE DAWID loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}'. \
                format(np.mean(loss_baseline_list_dawid), cost_baseline, np.mean(fp_d), np.mean(tp_d),
                       np.mean(rec_d), np.mean(pre_d))
            print 'BASELINE MV    loss: {:1.2f}, price: {:1.2f}, fp_rate: {:1.2f}, tp_rate: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}'. \
                format(np.mean(loss_baseline_list_mv), cost_baseline, np.mean(fp_mv), np.mean(tp_mv),
                       np.mean(rec_mv), np.mean(pre_mv))
            print '---------------------'

            data.append([Nt, J, cr, np.mean(loss_baseline_list), np.std(loss_baseline_list),
                         np.mean(fp_b), np.mean(tp_b), cost_baseline, 0., 'dong',
                         np.mean(rec_b), np.mean(pre_b)])
            data.append([Nt, J, cr, np.mean(loss_baseline_list_dawid), np.std(loss_baseline_list_dawid),
                         np.mean(fp_d), np.mean(tp_d), cost_baseline, 0., 'dawid',
                         np.mean(rec_d), np.mean(pre_d)])
            data.append([Nt, J, cr, np.mean(loss_baseline_list_mv), np.std(loss_baseline_list_mv),
                         np.mean(fp_mv), np.mean(tp_mv), cost_baseline, 0., 'mv',
                         np.mean(rec_mv), np.mean(pre_mv)])
    pd.DataFrame(data, columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'FPR', 'TPR',
                                'price_mean', 'price_std', 'alg', 'recall', 'precision']). \
                                to_csv('output/data/dong_dawid_mv.csv', index=False)
