from helpers.utils import classify_papers, compute_metrics
from fusion_algorithms.dawid_skene import dawid_skene
from fusion_algorithms.mv import majority_voting
from fusion_algorithms.algorithms_utils import input_adapter


def baseline_dong(responses, criteria_num, n_papers, papers_page, J, GT, cost):
    classified_papers = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)
    loss, fp_rate, fn_rate, recall, precision = compute_metrics(classified_papers, GT, cost, criteria_num)
    return loss, fp_rate, fn_rate, recall, precision


def baseline_dawid(responses, criteria_num, n_papers, GT, cost):
    values_prob = dawid_skene(responses, tol=0.001, max_iter=50)
    classified_papers = []
    exclusion_trsh = cost / (cost + 1.)
    for paper_id in range(n_papers):
        p_inclusion = 1.
        for e_paper_id in range(criteria_num):
            p_inclusion *= values_prob[paper_id * criteria_num + e_paper_id][0]
        p_exclusion = 1 - p_inclusion
        classified_papers.append(0) if p_exclusion > exclusion_trsh else classified_papers.append(1)
    loss, fp_rate, fn_rate, recall, precision = compute_metrics(classified_papers, GT, cost, criteria_num)
    return loss, fp_rate, fn_rate, recall, precision


def baseline_mv(responses, criteria_num, n_papers, papers_page, J, GT, cost):
    Psi = input_adapter(responses)
    p = majority_voting(Psi)
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.iteritems():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)
    classified_papers = []
    exclusion_trsh = cost / (cost + 1.)
    for paper_id in range(n_papers):
        p_inclusion = 1.
        for e_paper_id in range(criteria_num):
            p_inclusion *= values_prob[paper_id * criteria_num + e_paper_id][0]
        p_exclusion = 1 - p_inclusion
        classified_papers.append(0) if p_exclusion > exclusion_trsh else classified_papers.append(1)
    loss, fp_rate, fn_rate, recall, precision = compute_metrics(classified_papers, GT, cost, criteria_num)
    return loss, fp_rate, fn_rate, recall, precision
