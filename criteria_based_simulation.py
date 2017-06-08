from generator import generate_responses_gt
from fusion_algorithms.dawid_skene import dawid_skene
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization
from utils import run_quiz_criteria_confm, aggregate_papers, estimate_loss
import pandas as pd


def get_loss_dawid(responses):
    values_prob = dawid_skene(responses, tol=0.001, max_iter=50)
    papers_prob_in = aggregate_papers(n_papers, criteria_num, values_prob)
    loss = estimate_loss(papers_prob_in, cost)
    return loss


def get_loss_dong(responses):
    Psi = input_adapter(responses)
    N = (n_papers / papers_page) * J
    p = expectation_maximization(N, n_papers*criteria_num, Psi)[1]
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.iteritems():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)
    papers_prob_in = aggregate_papers(n_papers, criteria_num, values_prob)
    loss = estimate_loss(papers_prob_in, cost)
    return loss


if __name__ == '__main__':
    z = 0.3
    cost = 5
    n_papers = 500
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    data = []
    for Nt in range(1, 11, 1):
        print Nt
        acc = run_quiz_criteria_confm(Nt, z, criteria_difficulty)
        for J in [2, 3, 5, 10]:
            responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page, J, acc, criteria_difficulty)
            loss_dawid = get_loss_dawid(responses)
            loss_dong = get_loss_dong(responses)
            data.append([Nt, J, cost, loss_dawid, 'dawid'])
            data.append([Nt, J, cost, loss_dong, 'dong'])
    pd.DataFrame(data, columns=['Nt', 'J', 'cost', 'loss', 'alg']). \
        to_csv('output/data/loss_tests_cr5.csv', index=False)
