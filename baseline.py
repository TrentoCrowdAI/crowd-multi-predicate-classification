from helpers.utils import classify_papers, compute_metrics


def baseline(responses, criteria_num, n_papers, cost, GT):
    classified_papers = classify_papers(responses, criteria_num, n_papers, cost)
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(classified_papers, GT, cost, criteria_num)
    # Count votes
    votes_count = 0
    for l in responses:
        votes_count += len(l)
    price_per_paper = float(votes_count) / n_papers
    return loss, recall, precision, f_beta, price_per_paper
