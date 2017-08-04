from generator import generate_responses_gt
from helpers.method_2 import classify_papers_baseline, generate_responses, \
    update_v_count, assign_criteria, classify_papers, update_cr_power
from helpers.utils import compute_metrics, estimate_cr_power_dif, get_roc_points
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization
import operator


def do_first_round(n_papers, criteria_num, papers_worker, J, lr, GT,
                   criteria_power, acc, criteria_difficulty, values_count):
    # generate responses
    responses = generate_responses_gt(n_papers, criteria_power, papers_worker,
                                      J, acc, criteria_difficulty, GT)
    # aggregate responses
    Psi = input_adapter(responses)
    N = (n_papers / papers_worker) * J
    p = expectation_maximization(N, n_papers * criteria_num, Psi)[1]
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.iteritems():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    power_cr_list, acc_cr_list = estimate_cr_power_dif(responses, criteria_num, n_papers, papers_worker, J)
    classified_papers, rest_p_ids, papers_prob = classify_papers_baseline(range(n_papers), criteria_num, values_prob, lr)
    # count value counts
    for key in range(n_papers*criteria_num):
        cr_resp = responses[key]
        for v in cr_resp.values():
            values_count[key][v[0]] += 1
    return classified_papers, rest_p_ids, power_cr_list, acc_cr_list, papers_prob


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


def sm_run(criteria_num, n_papers, papers_worker, J, lr, Nt, acc,
           criteria_power, criteria_difficulty, GT, fr_p_part):
    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(n_papers*criteria_num)]

    # Baseline round
    # in% papers
    fr_n_papers = int(n_papers * fr_p_part)
    criteria_count = (Nt + papers_worker * criteria_num) * J * fr_n_papers / papers_worker
    first_round_res = do_first_round(fr_n_papers, criteria_num, papers_worker, J, lr, GT,
                                     criteria_power, acc, criteria_difficulty, values_count)
    classified_papers_fr, rest_p_ids, power_cr_list, acc_cr_list, papers_prob = first_round_res
    classified_papers = dict(zip(range(n_papers), [1]*n_papers))
    classified_papers.update(classified_papers_fr)
    rest_p_ids = rest_p_ids + range(fr_n_papers, n_papers)

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
        classified_p_round, rest_p_ids, papers_prob_it = classify_papers(rest_p_ids, criteria_num, values_count,
                                                         p_thrs, acc_cr_list, power_cr_list)
        papers_prob.update(papers_prob_it)

        # update criteria power
        power_cr_list = update_cr_power(n_papers, criteria_num, acc_cr_list, power_cr_list, values_count)

        # print len(rest_p_ids)
        n_rest = len(rest_p_ids)
        break_list.append(n_rest)
        if break_list.count(n_rest) >= 5:
            break
        classified_papers.update(classified_p_round)
    #     TO DO: rest papers
    classified_papers = [classified_papers[p_id] for p_id in sorted(classified_papers.keys())]
    sorted_papers_prob = sorted(papers_prob.items(), key=operator.itemgetter(1), reverse=True)
    roc_points = get_roc_points(GT, sorted_papers_prob, classified_papers, criteria_num)
    return roc_points
