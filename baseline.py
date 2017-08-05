from helpers.utils import classify_papers, prepare_roc_data


def baseline(responses, criteria_num, n_papers, papers_page, J, GT, cost):
    classified_papers, papers_prob = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)
    N, P, papers_prob = prepare_roc_data(GT, papers_prob, criteria_num)
    return N, P, papers_prob
