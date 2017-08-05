import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm, get_roc_points
from baseline import baseline
from m_run import m_run
from sm_run import sm_run

if __name__ == '__main__':
    z = 0.3
    lr = 5
    n_papers = 1000
    papers_page = 10
    # criteria_power = [0.14, 0.14, 0.28, 0.42]
    # theta = 0.5
    criteria_power = [0.09, 0.09, 0.18, 0.27]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    fr_p_part = 0.05
    data = []
    # for Nt in range(1, 11, 1):
    #     for J in [3, 5, 10]:
    for Nt in [4]:
        for J in [5]:
            print 'Nt: {}. J: {}'.format(Nt, J)
            cost_baseline = (Nt + papers_page * criteria_num) * J / float(papers_page)
            Nb, Pb, Nm, Pm, Nsm, Psm = 0., 0., 0., 0., 0., 0.
            probs_b_list, probs_m_list, probs_sm_list = [], [], []
            for _ in range(30):
                # quiz, generation responses
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                # baseline
                N_b, P_b, probs_b = baseline(responses, criteria_num, n_papers, papers_page, J, GT, lr)
                Nb += N_b
                Pb += P_b
                probs_b_list += probs_b

                # m-run
                N_m, P_m, probs_m = m_run(criteria_num, n_papers, papers_page, J, lr, Nt, acc,
                                          criteria_power, criteria_difficulty, GT, fr_p_part)
                Nm += N_m
                Pm += P_m
                probs_m_list += probs_m

                # sm-run
                N_sm, P_sm, probs_sm = sm_run(criteria_num, n_papers, papers_page, J, lr, Nt, acc,
                                              criteria_power, criteria_difficulty, GT, fr_p_part)
                Nsm += N_sm
                Psm += P_sm
                probs_sm_list += probs_sm

            sorted_probs_b = sorted(probs_b_list, key=lambda x: x[1], reverse=True)
            roc_points_b = get_roc_points(Nb, Pb, sorted_probs_b)

            sorted_probs_m = sorted(probs_m_list, key=lambda x: x[1], reverse=True)
            roc_points_m = get_roc_points(Nm, Pm, sorted_probs_m)

            sorted_probs_sm = sorted(probs_sm_list, key=lambda x: x[1], reverse=True)
            roc_points_sm = get_roc_points(Nsm, Psm, sorted_probs_sm)

            # np_df = pd.DataFrame({'Nb': [Nb], 'Pb': [Pb],
            #                       'Nm': [Nm], 'Pm': [Pm],
            #                       'Nsm': [Nsm], 'Psm': [Psm]})
            roc_b_df = pd.DataFrame({'x_b': roc_points_b[0], 'y_b': roc_points_b[1]})
            roc_m_df = pd.DataFrame({'x_m': roc_points_m[0], 'y_m': roc_points_m[1]})
            roc_sm_df = pd.DataFrame({'x_sm': roc_points_sm[0], 'y_sm': roc_points_sm[1]})
            data = pd.concat([roc_b_df, roc_m_df, roc_sm_df], ignore_index=True)
            data.to_csv('output/data/roc_curves.csv', index=False)

