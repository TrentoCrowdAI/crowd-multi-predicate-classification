from helpers.method_2 import classify_papers_baseline, \
    update_v_count, assign_criteria, classify_papers, update_cr_power
from helpers.utils import compute_metrics, estimate_cr_power_dif
from fusion_algorithms.em import expectation_maximization
import numpy as np


def do_first_round(responses, criteria_num, n_papers, lr, values_count):
    N = -1
    # transform data
    workers_ids = {}
    for i in range(n_papers * criteria_num):
        for j, c_data in enumerate(responses[i]):
            if c_data[0] not in workers_ids.keys():
                N += 1
                workers_ids[c_data[0]] = N
                responses[i][j] = (N, c_data[1])
            else:
                responses[i][j] = (workers_ids[c_data[0]], c_data[1])
    N += 1
    p = expectation_maximization(N, n_papers * criteria_num, responses)[1]
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.iteritems():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    power_cr_list, acc_cr_list = estimate_cr_power_dif(responses, criteria_num, n_papers)
    classified_papers, rest_p_ids = classify_papers_baseline(range(n_papers), criteria_num, values_prob, lr)
    # count value counts
    for key in range(n_papers*criteria_num):
        cr_resp = responses[key]
        for _, v in cr_resp:
            values_count[key][v] += 1
    return classified_papers, rest_p_ids, power_cr_list, acc_cr_list


def do_round(c_votes, rest_p_ids, criteria_num, cr_assigned, GT, criteria_accuracy):
    counter = 0
    responses = []
    for paper_id, cr in zip(rest_p_ids, cr_assigned):
        if c_votes[paper_id * criteria_num + cr]:
            vote = c_votes[paper_id * criteria_num + cr].pop(0)[1]
        else:
            gt = GT[paper_id * criteria_num + cr]
            mean = criteria_accuracy[cr][0]
            sigma = (criteria_accuracy[cr][2] - criteria_accuracy[cr][0]) / 2
            acc = np.random.normal(mean, sigma, 1)[0]
            if acc < 0.5:
                acc = 0.5
            if acc > 1.:
                acc = 0.95
            vote = np.random.binomial(gt, acc, 1)[0]
            counter += 1
        responses.append(vote)
    return responses, counter


def sm_run(c_votes, criteria_num, n_papers, lr, GT, fr_p_part, criteria_accuracy):
    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(n_papers*criteria_num)]
    # Baseline round
    # in% papers
    fr_n_papers = int(n_papers * fr_p_part)
    responses_fround = c_votes[: fr_n_papers * criteria_num]
    # Count votes 1st round
    votes_count = 0
    syn_v_counter = 0
    for l in responses_fround:
        votes_count += len(l)
        for i in l:
            if i[0] == 3333:
                syn_v_counter += 1
    first_round_res = do_first_round(responses_fround, criteria_num, fr_n_papers, lr, values_count)
    classified_papers_fr, rest_p_ids, power_cr_list, acc_cr_list = first_round_res
    # acc_cr_list = [criteria_accuracy[0][0], criteria_accuracy[1][0]]
    # acc_cr_list = [acc * 0.9 for acc in acc_cr_list]
    classified_papers = dict(zip(range(n_papers), [1]*n_papers))
    classified_papers.update(classified_papers_fr)
    rest_p_ids = rest_p_ids + range(fr_n_papers, n_papers)
    # Do Multi rounds
    while len(rest_p_ids) != 0:
        votes_count += len(rest_p_ids)

        # criteria_count += len(rest_p_ids)
        cr_assigned, in_papers_ids, rest_p_ids = assign_criteria(rest_p_ids, criteria_num, values_count, power_cr_list,
                                                                 acc_cr_list)
        for i in in_papers_ids:
            classified_papers[i] = 1

        responses, round_syn_votes = do_round(c_votes, rest_p_ids, criteria_num, cr_assigned, GT, criteria_accuracy)
        syn_v_counter += round_syn_votes
        # update values_count
        update_v_count(values_count, criteria_num, cr_assigned, responses, rest_p_ids)

        # update criteria power
        power_cr_list = update_cr_power(n_papers, criteria_num, acc_cr_list, power_cr_list, values_count)

        # classify papers
        classified_p_round, rest_p_ids = classify_papers(rest_p_ids, criteria_num, values_count,
                                                         p_thrs, acc_cr_list, power_cr_list)
        classified_papers.update(classified_p_round)
    classified_papers = [classified_papers[p_id] for p_id in sorted(classified_papers.keys())]
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(classified_papers, GT, lr, criteria_num)
    price_per_paper = float(votes_count) / n_papers
    syn_votes_prop = syn_v_counter / float(votes_count)
    return loss, recall, precision, f_beta, price_per_paper, syn_votes_prop
