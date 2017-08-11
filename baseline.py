from helpers.utils import classify_papers, compute_metrics


def baseline(responses, criteria_num, n_papers, cost, GT):
    classified_papers = classify_papers(responses, criteria_num, n_papers, cost)
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(classified_papers, GT, cost, criteria_num)
    return loss, fp_rate, fn_rate, recall, precision, f_beta
