from generator import generate_responses_gt
from helpers.method_2 import classify_papers_baseline, generate_responses, \
    update_v_count, assign_criteria, classify_papers, update_cr_power
from helpers.utils import compute_metrics, estimate_cr_power_dif
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


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


def do_round(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
    # generate responses
    n = len(papers_ids)
    papers_ids_rest1 = papers_ids[:n - n % papers_worker]
    papers_ids_rest2 = papers_ids[n - n % papers_worker:]
    responses_rest1 = generate_responses(GT, papers_ids_rest1, criteria_num,
                                         papers_worker, acc, criteria_difficulty,
                                         cr_assigned)
    responses_rest2 = generate_responses(GT, papers_ids_rest2, criteria_num,
                                         papers_worker, acc, criteria_difficulty,
                                         cr_assigned)
    responses = responses_rest1 + responses_rest2
    return responses


def sm_run(c_votes, criteria_num, n_papers, lr, GT, fr_p_part):
    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(n_papers*criteria_num)]

    # Baseline round
    # in% papers
    fr_n_papers = int(n_papers * fr_p_part)
    responses_fround = c_votes[: fr_n_papers * criteria_num]
    first_round_res = do_first_round(responses_fround, criteria_num, fr_n_papers, lr, values_count)

    classified_papers_fr, rest_p_ids, power_cr_list, acc_cr_list = first_round_res
    classified_papers = dict(zip(range(n_papers), [1]*n_papers))
    classified_papers.update(classified_papers_fr)
    rest_p_ids = rest_p_ids + range(fr_n_papers, n_papers)
    # TO DO
    # Do Multi rounds
    break_list = []
    while len(rest_p_ids) != 0:
        # print len(rest_p_ids)

        criteria_count += len(rest_p_ids)
        cr_assigned = assign_criteria(rest_p_ids, criteria_num, values_count, power_cr_list, acc_cr_list)

        responses = do_round(GT, rest_p_ids, criteria_num, papers_worker*criteria_num,
                             acc, criteria_difficulty, cr_assigned)
        # update values_count
        update_v_count(values_count, criteria_num, cr_assigned, responses, rest_p_ids)

        # classify papers
        classified_p_round, rest_p_ids = classify_papers(rest_p_ids, criteria_num, values_count,
                                                                              p_thrs, acc_cr_list, power_cr_list)

        # update criteria power
        power_cr_list = update_cr_power(n_papers, criteria_num, acc_cr_list, power_cr_list, values_count)

        # print len(rest_p_ids)
        n_rest = len(rest_p_ids)
        break_list.append(n_rest)
        if break_list.count(n_rest) >= 5:
            break
        classified_papers.update(classified_p_round)
    classified_papers = [classified_papers[p_id] for p_id in sorted(classified_papers.keys())]
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(classified_papers, GT, lr, criteria_num)
    price_per_paper = float(criteria_count) / n_papers
    return loss, price_per_paper, fp_rate, fn_rate, recall, precision, f_beta
