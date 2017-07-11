import numpy as np

from generator import generate_responses_gt
from helpers.method_2 import classify_papers, generate_responses, \
    update_v_prob, assign_criteria
from helpers.utils import get_actual_loss
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def do_first_round(n_papers, criteria_num, papers_worker, J, lr, GT,
                   criteria_power, acc, criteria_difficulty, values_prob):
    N = (n_papers / papers_worker) * J   # N workers
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
                values_prob[len(GT_cr)*cr+e_id][e_v] = e_p
            p_out_list.append(values_prob[len(GT_cr)*cr+e_id][1])
        power_cr_list.append(np.mean(p_out_list))
        acc_cr_list.append(np.mean(a))

    #  initialise values_prob with priors=criteria power
    for cr in range(criteria_num):
        for cr_ind in range(n_papers*criteria_num+cr, len(values_prob), criteria_num):
            values_prob[cr_ind][0] = 1 - power_cr_list[cr]
            values_prob[cr_ind][1] = power_cr_list[cr]

    classified_papers, rest_p_ids = classify_papers(range(n_papers), criteria_num, values_prob, lr)
    return classified_papers, rest_p_ids, power_cr_list, acc_cr_list


def do_round(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
    # generate responses
    n = len(papers_ids)
    papers_ids_rest1 = papers_ids[:n - n % papers_worker]
    papers_ids_rest2 = papers_ids[n - n % papers_worker:]
    responses_rest1 = generate_responses(GT, papers_ids_rest1, criteria_num,
                                         papers_worker, acc, criteria_difficulty,
                                         cr_assigned)
    responses_rest2 = generate_responses(GT, papers_ids_rest2, criteria_num,
                                         papers_worker, acc, criteria_difficulty,
                                         cr_assigned)
    responses = responses_rest1 + responses_rest2
    return responses


def get_loss_cost_smrun(criteria_num, n_papers, papers_worker, J, lr, Nt,
                        acc, criteria_power, criteria_difficulty, GT):
    # initialization
    values_prob = [[0., 0.] for _ in range(n_papers*criteria_num)]

    # Baseline round
    # 10% papers
    criteria_count = (Nt + papers_worker * criteria_num) * J * (n_papers / 10) / papers_worker
    first_round_res = do_first_round(n_papers/10, criteria_num, papers_worker, J, 30, GT,
                                     criteria_power, acc, criteria_difficulty, values_prob)

    classified_papers = dict(zip(range(n_papers), [1]*n_papers))
    classified_papers_fr, rest_p_ids, power_cr_list, acc_cr_list = first_round_res
    classified_papers.update(classified_papers_fr)

    rest_p_ids = rest_p_ids + range(n_papers / 10, n_papers)
    # print len(rest_p_ids)
    # Do Multi rounds
    break_list = []
    while True:
        criteria_count += len(rest_p_ids)
        cr_assigned = assign_criteria(rest_p_ids, criteria_num, values_prob, acc_cr_list)

        responses = do_round(GT, rest_p_ids, criteria_num, papers_worker*criteria_num,
                             acc, criteria_difficulty, cr_assigned)
        # update values_prob
        update_v_prob(values_prob, responses, rest_p_ids, cr_assigned, criteria_num, acc_cr_list)

        # classify papers
        classified_p_round, rest_p_ids = classify_papers(rest_p_ids, criteria_num, values_prob, 30)

        # print len(rest_p_ids)
        n_rest = len(rest_p_ids)
        break_list.append(n_rest)
        if break_list.count(n_rest) >= 20:
            break

        classified_papers.update(classified_p_round)
    # TO DO
    classified_papers = [classified_papers[p_id] for p_id in sorted(classified_papers.keys())]
    loss = get_actual_loss(classified_papers, GT, lr, criteria_num)
    price_per_paper = float(criteria_count) / n_papers
    return loss, price_per_paper
