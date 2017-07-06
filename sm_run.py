import numpy as np

from generator import generate_responses_gt
from helpers.method_2 import classify_papers
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def do_first_round(n_papers, criteria_num, papers_worker, J, lr, GT,
                   criteria_power, acc, criteria_difficulty, values_prob):
    # n workers
    N = (n_papers / papers_worker) * J

    power_cr_list = []
    acc_cr_list = []
    for cr in range(criteria_num):
        # generate responses
        GT_cr = [GT[p_id * criteria_num + cr] for p_id in range(n_papers)]
        responses_cr = generate_responses_gt(n_papers, [criteria_power[cr]], papers_worker,
                                             J, acc, [criteria_difficulty[cr]], GT_cr)
        # aggregate responses
        Psi = input_adapter(responses_cr)
        a, p = expectation_maximization(N, n_papers, Psi)

        # update values_prob, copmute cr_power, cr_accuracy
        p_out_list = []
        for e_id, e in enumerate(p):
            for e_v, e_p in e.iteritems():
                values_prob[e_id][e_v] = e_p
            p_out_list.append(values_prob[e_id][1])
        power_cr_list.append(np.mean(p_out_list))
        acc_cr_list.append(np.mean(a))

    classified_papers, classified_papers_ids, rest_papers_ids = \
        classify_papers(range(n_papers), criteria_num, values_prob, lr)
    pass

    # for paper_id in range(n_papers):
    #     p_exclusion = 1.
    #     for e_paper_id in range(criteria_num):
    #         p_exclusion *= 1 - values_prob[paper_id * criteria_num + e_paper_id][1]
    #     p_exclusion = 1 - p_exclusion
    #     if p_exclusion > trsh:
    #         classified_papers.append(0)
    #         classified_papers_ids.append(paper_id)
    #     elif 1 - p_exclusion > trsh:
    #         classified_papers.append(1)
    #         classified_papers_ids.append(paper_id)

    return None


def get_loss_cost_smrun(criteria_num, n_papers, papers_worker, J, lr, Nt,
                        acc, criteria_power, criteria_difficulty, GT):
    # initialization
    values_prob = [[0., 0.] for _ in range(n_papers*criteria_num)]

    # Baseline round
    # 10% papers
    criteria_count = (Nt + papers_worker * criteria_num) * J * (n_papers / 10) / papers_worker
    first_round_res = do_first_round(n_papers/10, criteria_num, papers_worker, J, lr, GT,
                                     criteria_power, acc, criteria_difficulty, values_prob)


    # Do Multi rounds


    pass
