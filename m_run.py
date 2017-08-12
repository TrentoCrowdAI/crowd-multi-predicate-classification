import numpy as np

from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization
from generator import generate_responses_gt
from helpers.utils import classify_papers, compute_metrics, estimate_cr_power_dif


def get_best_cr_order(responses, criteria_num, n_papers, papers_page, J):
    cr_power, cr_accuracy = estimate_cr_power_dif(responses, criteria_num, n_papers, papers_page, J)
    # TO DO
    # estimate_cr_order(cr_power, cr_accuracy)
    best_cr_order = range(criteria_num)
    return best_cr_order


def first_round(responses, criteria_num, n_papers, cost):
    classified_papers = classify_papers(responses, criteria_num, n_papers, cost)
    # best_cr_order = range(criteria_num)[::-1]
    # best_cr_order = [1, 2, 0]
    best_cr_order = [1, 0]
    return classified_papers, best_cr_order


def do_round(c_votes, cr, papers_ids_rest, criteria_num, cost):
    n_papers = len(papers_ids_rest)
    responses_round = [c_votes[p_id*criteria_num+cr] for p_id in papers_ids_rest]
    classified_papers = classify_papers(responses_round, 1, n_papers, cost)
    classified_papers = zip(papers_ids_rest, classified_papers)
    # Count votes
    n_votes = 0
    for l in responses_round:
        n_votes += len(l)
    return classified_papers, n_votes


def m_run(c_votes, criteria_num, n_papers, cost, GT, fr_p_part):
    # first round responses
    fr_n_papers = int(n_papers*fr_p_part)
    responses_fround = c_votes[: fr_n_papers*criteria_num]
    classified_papers_fround, criteria_order = first_round(responses_fround, criteria_num, fr_n_papers, cost)
    # Count votes 1st round
    votes_count = 0
    for l in responses_fround:
        votes_count += len(l)
    # Do Multi rounds
    papers_ids_rest = range(fr_n_papers, n_papers, 1)
    classified_papers = classified_papers_fround + [1 for _ in papers_ids_rest]

    for cr in criteria_order:
        if papers_ids_rest:
            classified_papers_cr, n_votes = do_round(c_votes, cr, papers_ids_rest, criteria_num, cost)
            votes_count += n_votes
        papers_ids_rest = []
        for p_id, p_cr in classified_papers_cr:
            if p_cr:
                papers_ids_rest.append(p_id)
            else:
                classified_papers[p_id] = 0
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(classified_papers, GT, cost, criteria_num)
    price_per_paper = float(votes_count) / n_papers
    return loss, recall, precision, f_beta, price_per_paper
