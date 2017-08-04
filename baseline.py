from helpers.utils import classify_papers, get_roc_points
import operator


def baseline(responses, criteria_num, n_papers, papers_page, J, GT, cost):
    classified_papers, papers_prob = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)
    sorted_papers_prob = sorted(papers_prob.items(), key=operator.itemgetter(1), reverse=True)
    roc_points = get_roc_points(GT, sorted_papers_prob, classified_papers, criteria_num)
    return roc_points
