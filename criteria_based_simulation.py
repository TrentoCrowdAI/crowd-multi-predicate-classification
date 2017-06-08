from generator import generate_responses_gt
from fusion_algorithms.dawid_skene import dawid_skene
from utils import run_quiz_criteria_confm, aggregate_papers, estimate_loss
import pandas as pd


if __name__ == '__main__':
    z = 0.3
    cost = 2
    n_papers = 1000
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    data = []
    for Nt in range(1, 11, 1):
        acc = run_quiz_criteria_confm(Nt, z, criteria_difficulty)
        for J in [2, 3, 5, 10]:
            responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page, J, acc, criteria_difficulty)
            values_prob = dawid_skene(responses, tol=0.00001, max_iter=100)  # TO DO max_iter=100
            papers_prob_in = aggregate_papers(n_papers, criteria_num, values_prob)
            loss = estimate_loss(papers_prob_in, cost)
            data.append([Nt, J, loss, cost])
    pd.DataFrame(data, columns=['Nt', 'J', 'loss', 'cost']). \
        to_csv('output/data/loss_tests_theta03.csv', index=False)
