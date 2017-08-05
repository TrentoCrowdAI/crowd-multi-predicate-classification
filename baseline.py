from helpers.utils import classify_papers, prepare_roc_data, get_roc_points
import operator


def baseline(responses, criteria_num, n_papers, papers_page, J, GT, cost):
    classified_papers, papers_prob = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)
    N, P, papers_prob = prepare_roc_data(GT, classified_papers, papers_prob, criteria_num)
    return N, P, classified_papers
