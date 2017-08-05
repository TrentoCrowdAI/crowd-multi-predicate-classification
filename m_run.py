import numpy as np

from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization
from generator import generate_responses_gt
from helpers.utils import classify_papers, classify_papers_m, compute_metrics, estimate_cr_power_dif, get_roc_points
import operator


def get_best_cr_order(responses, criteria_num, n_papers, papers_page, J):
    cr_power, cr_accuracy = estimate_cr_power_dif(responses, criteria_num, n_papers, papers_page, J)
    # TO DO
    # estimate_cr_order(cr_power, cr_accuracy)
    best_cr_order = range(criteria_num)
    return best_cr_order


def first_round(responses, criteria_num, n_papers, papers_page, J, cost):
    classified_papers, papers_prob = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)
    best_cr_order = get_best_cr_order(responses, criteria_num, n_papers, papers_page, J)
    return classified_papers, best_cr_order, papers_prob


def do_round(GT, cr, papers_ids_rest, criteria_num, papers_worker, J,
             cost, acc, criteria_power, criteria_difficulty):
    n_papers = len(papers_ids_rest)
    GT_round = [GT[p_id*criteria_num+cr] for ind, p_id in enumerate(papers_ids_rest)]
    responses_round = generate_responses_gt(n_papers, [criteria_power[cr]], papers_worker,
                                            J, acc, [criteria_difficulty[cr]], GT_round)
    classified_papers, papers_prob = classify_papers_m(n_papers, 1, responses_round, papers_worker, J, cost)
    papers_prob = dict(zip(papers_ids_rest, papers_prob))
    classified_papers = zip(papers_ids_rest, classified_papers)
    return classified_papers, papers_prob


def m_run(criteria_num, n_papers, papers_page, J, cost, Nt, acc,
          criteria_power, criteria_difficulty, GT, fr_p_part):
    # first round responses
    fr_n_papers = int(n_papers*fr_p_part)
    criteria_count = (Nt + papers_page * criteria_num) * J * fr_n_papers / papers_page
    GT_fround = GT[: fr_n_papers*criteria_num]
    responses_fround = generate_responses_gt(fr_n_papers, criteria_power, papers_page,
                                             J, acc, criteria_difficulty, GT_fround)
    classified_papers_fround, best_cr_order, papers_prob = first_round(responses_fround, criteria_num,
                                                                       fr_n_papers, papers_page, J, cost)
    # Do Multi rounds
    papers_ids_rest = range(fr_n_papers, n_papers, 1)
    classified_papers = classified_papers_fround + [1 for _ in papers_ids_rest]

    papers_worker = papers_page * criteria_num
    for cr in best_cr_order:
        n_rest = len(papers_ids_rest)
        papers_ids_rest1 = papers_ids_rest[:n_rest - n_rest % papers_worker]
        papers_ids_rest2 = papers_ids_rest[n_rest - n_rest % papers_worker:]
        if papers_ids_rest1:
            criteria_count += (Nt + papers_worker) * J * len(papers_ids_rest1) / float(papers_worker)
            classified_papers_cr, papers_prob_it1 = do_round(GT, cr, papers_ids_rest1, criteria_num, papers_worker, J,
                                                             cost, acc, criteria_power, criteria_difficulty)
            papers_prob.update(papers_prob_it1)
        # check if n_papers_rest % papers_page != 0 then run an additional round
        if papers_ids_rest2:
            criteria_count += (Nt + len(papers_ids_rest2)) * J
            classified_papers_cr2, papers_prob_it2 = do_round(GT, cr, papers_ids_rest2, criteria_num, n_rest % papers_worker, J,
                                                              cost, acc, criteria_power, criteria_difficulty)
            classified_papers_cr += classified_papers_cr2
            papers_prob.update(papers_prob_it2)
        papers_ids_rest = []
        for p_id, p_cr in classified_papers_cr:
            if p_cr:
                papers_ids_rest.append(p_id)
            else:
                classified_papers[p_id] = 0
    sorted_papers_prob = sorted(papers_prob.items(), key=operator.itemgetter(1), reverse=True)
    roc_points = get_roc_points(GT, sorted_papers_prob, classified_papers, criteria_num)
    return roc_points
