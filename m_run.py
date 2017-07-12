import numpy as np

from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization
from generator import generate_responses_gt
from helpers.utils import classify_papers, get_actual_loss


def get_best_cr_order(responses, criteria_num, n_papers, papers_page, J):
    cr_power, cr_accuracy = estimate_cr_power_dif(responses, criteria_num, n_papers, papers_page, J)
    # TO DO
    # estimate_cr_order(cr_power, cr_accuracy)
    best_cr_order = range(criteria_num)
    return best_cr_order


def estimate_cr_power_dif(responses, criteria_num, n_papers, papers_page, J):
    Psi = input_adapter(responses)
    N = (n_papers / papers_page) * J

    cr_power = []
    cr_accuracy = []
    for cr in range(criteria_num):
        cr_responses = Psi[cr::criteria_num]
        acc_list, p_cr = expectation_maximization(N, n_papers, cr_responses)
        acc_cr = np.mean(acc_list)
        pow_cr = 0.
        for e in p_cr:
            e_prob = [0., 0.]
            for e_id, e_p in e.iteritems():
                e_prob[e_id] = e_p
            pow_cr += e_prob[1]
        pow_cr /= float(n_papers)
        cr_power.append(pow_cr)
        cr_accuracy.append(acc_cr)

    return cr_power, cr_accuracy


def first_round(responses, criteria_num, n_papers, papers_page, J, cost):
    classified_papers = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)
    best_cr_order = get_best_cr_order(responses, criteria_num, n_papers, papers_page, J)
    return classified_papers, best_cr_order


def do_round(GT, cr, papers_ids_rest, criteria_num, papers_worker, J,
             cost, acc, criteria_power, criteria_difficulty):
    n_papers = len(papers_ids_rest)
    GT_round = [GT[p_id*criteria_num+cr] for ind, p_id in enumerate(papers_ids_rest)]
    responses_round = generate_responses_gt(n_papers, [criteria_power[cr]], papers_worker,
                                            J, acc, [criteria_difficulty[cr]], GT_round)
    classified_papers = zip(papers_ids_rest, classify_papers(n_papers, 1, responses_round, papers_worker, J, cost))
    return classified_papers


def get_loss_cost_mrun(criteria_num, n_papers, papers_page, J, cost, Nt,
                       acc, criteria_power, criteria_difficulty, GT):
    # first round responses
    # 10% papers
    criteria_count = (Nt + papers_page * criteria_num) * J * (n_papers/10) / papers_page
    GT_fround = GT[: int(n_papers*criteria_num*0.1)]
    responses_fround = generate_responses_gt(n_papers/10, criteria_power, papers_page,
                                             J, acc, criteria_difficulty, GT_fround)
    classified_papers_fround, best_cr_order = first_round(responses_fround, criteria_num,
                                                          n_papers/10, papers_page, J, cost)
    # Do Multi rounds
    papers_ids_rest = range(n_papers/10, n_papers, 1)
    classified_papers = classified_papers_fround + [1 for _ in papers_ids_rest]

    papers_worker = papers_page * criteria_num
    for cr in best_cr_order:
        n_rest = len(papers_ids_rest)
        papers_ids_rest1 = papers_ids_rest[:n_rest - n_rest % papers_worker]
        papers_ids_rest2 = papers_ids_rest[n_rest - n_rest % papers_worker:]
        if papers_ids_rest1:
            criteria_count += (Nt + papers_worker) * J * len(papers_ids_rest1) / float(papers_worker)
            classified_papers_cr = do_round(GT, cr, papers_ids_rest1, criteria_num, papers_worker, J,
                                            cost, acc, criteria_power, criteria_difficulty)
        # check if n_papers_rest % papers_page != 0 then run an additional round
        if papers_ids_rest2:
            criteria_count += (Nt + len(papers_ids_rest2)) * J
            classified_papers_cr += do_round(GT, cr, papers_ids_rest2, criteria_num, n_rest % papers_worker, J,
                                             cost, acc, criteria_power, criteria_difficulty)
        papers_ids_rest = []
        for p_id, p_cr in classified_papers_cr:
            if p_cr:
                papers_ids_rest.append(p_id)
            else:
                classified_papers[p_id] = 0
    loss, fp_rate, fn_rate, recall, precision = get_actual_loss(classified_papers, GT, cost, criteria_num)
    cost = criteria_count / float(n_papers)
    return loss, cost, fp_rate, fn_rate, recall, precision
