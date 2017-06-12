from generator import generate_responses_gt
from fusion_algorithms.dawid_skene import dawid_skene
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization
from utils import run_quiz_criteria_confm, aggregate_papers,\
    estimate_loss, classify_papers, get_actual_loss
import pandas as pd
import numpy as np


def get_loss_dawid(responses):
    values_prob = dawid_skene(responses, tol=0.001, max_iter=50)
    # papers_prob_in = aggregate_papers(n_papers, criteria_num, values_prob)
    # loss = estimate_loss(papers_prob_in, cost)
    # get actual loss
    classified_papers = classify_papers(n_papers, criteria_num, values_prob, cost)
    loss = get_actual_loss(classified_papers, GT, cost, criteria_num)
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
    # papers_prob_in = aggregate_papers(n_papers, criteria_num, values_prob)
    # loss = estimate_loss(papers_prob_in, cost)
    # get actual loss
    classified_papers = classify_papers(n_papers, criteria_num, values_prob, cost)
    loss = get_actual_loss(classified_papers, GT, cost, criteria_num)
    return loss


def estimate_cr_power_dif(responses, criteria_num):
    values_prob = dawid_skene(responses, tol=0.001, max_iter=50)
    cr_difficulty = [0. for _ in range(criteria_num)]
    cr_power = [0. for _ in range(criteria_num)]
    n_papers_round = len(responses)/criteria_num
    for paper_id in range(n_papers_round):
        for e_id in range(criteria_num):
            prob_apply_e = values_prob[paper_id*criteria_num+e_id][1]
            cr_power[e_id] += prob_apply_e
            cr_difficulty[e_id] += abs(2*prob_apply_e-1)

    cr_difficulty = map(lambda x: 1-x/n_papers_round, cr_difficulty)
    cr_power = map(lambda x: x/n_papers_round, cr_power)
    return cr_power, cr_difficulty


def get_loss_mstrategy(responses):
    responses_part = {k: responses[k] for k in range(int(len(responses)*0.1))}
    cr_power_dif = estimate_cr_power_dif(responses_part, criteria_num)
    pass

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
        print Nt
        for J in [2, 3, 5, 10]:
            loss_dawid_list = []
            loss_dong_liss = []
            for _ in range(10):
                acc = run_quiz_criteria_confm(Nt, z, criteria_difficulty)
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)
                loss_dawid_list.append(get_loss_dawid(responses))
                loss_dong_liss.append(get_loss_dong(responses))
                # get_loss_mstrategy(responses)
            data.append([Nt, J, cost, np.mean(loss_dawid_list), np.std(loss_dawid_list), 'dawid'])
            data.append([Nt, J, cost, np.mean(loss_dong_liss), np.std(loss_dong_liss), 'dong'])
    pd.DataFrame(data, columns=['Nt', 'J', 'cost', 'loss_mean', 'loss_std', 'alg']). \
        to_csv('output/data/loss_tests_cr2_actual.csv', index=False)
