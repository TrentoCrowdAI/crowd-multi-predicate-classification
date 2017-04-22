'''
This script models the workflow of crowdflower platform
 for the job of scope tagging scientific papers by crowd users.
The experiment uses synthetic data generated in the current script.

Parameters for the crowdflower synthesizer:
    test_page - number of test questions per page,
    papers_page - number of papers(excluding test questions) per page,
    n_papers - set of papers to be tagged(excluding test questions), must: n_papers % papers_page = 0,
    price_row - price per one row in dollars,
    judgment_min - min number judgments per row,
    cheaters_prop - proportion of cheaters from whole population of workers

    OUTPUT: accuracy of results, spent budget, total number of pages being paid

'''

import random
import numpy as np


def get_metrics(gold_data, trusted_judgment, fp_cost, fn_cost):
    fp_count = 0
    fn_count = 0
    correct_count = 0
    total_rows = len(gold_data)
    p_count = sum([row[0] for row in gold_data])
    f_count = total_rows - p_count
    for gold, users_values in zip(gold_data, trusted_judgment):
        gold_value = gold[0]
        aggregated_value = max(set(users_values), key=users_values.count)
        # if the paper in scope
        if gold_value == aggregated_value:
            correct_count += 1
        else:
            if gold_value:
                fp_count += 1
            else:
                fn_count += 1
    acc = float(correct_count)/total_rows
    fp = float(fp_count)/p_count
    fn = float(fn_count)/f_count
    fp_lose = fp_count * fp_cost
    fn_lose = fn_count * fn_cost
    return [acc, fp, fn, fp_lose, fn_lose]


def pick_worker(user_prop, user_population):
    smart_ch = user_prop[1]
    t_worker = user_prop[2]
    p_val = np.random.uniform(0.0, 1.)
    if p_val < t_worker:
        worker_trust, worker_accuracy = user_population['worker'].pop(0)
        user_population['worker'].append((worker_trust, worker_accuracy))
    elif p_val >= t_worker and p_val < t_worker + smart_ch:
        worker_trust, worker_accuracy = user_population['smart_ch'].pop(0)
        user_population['smart_ch'].append((worker_trust, worker_accuracy))
    else:
        worker_trust, worker_accuracy = user_population['rand_ch'].pop(0)
        user_population['rand_ch'].append((worker_trust, worker_accuracy))
    return [worker_trust, worker_accuracy]


# get current worker's trust
def get_trust(w_page_judgment, gold_data, test_page, worker_trust, quiz_papers_n):
    correctly_tagged_tests = 0
    for row_id in w_page_judgment.keys()[:test_page]:
        if gold_data[row_id][0] == w_page_judgment[row_id]:
            correctly_tagged_tests += 1
    new_trust = (worker_trust*quiz_papers_n + correctly_tagged_tests)/(quiz_papers_n + test_page)
    return new_trust

'''
each element in 'trusted_workers_judgment' is judgments of users who passed tests questions,
indexes of the trusted_workers_judgment' present row id
'''
def do_round(trust_min, test_page, papers_page, n_papers, price_row, gold_data,
             judgment_min, user_prop, user_population, easy_add_acc, quiz_papers_n):
    pages_n = n_papers / papers_page
    rows_page = test_page+papers_page
    price_page = price_row*rows_page
    budget_spent = 0.
    trusted_judgment = [[] for _ in range(rows_page*pages_n)]
    # number of different types of workers after completing a page
    trusted_workers_n = 0
    untrusted_workers_n = 0
    worker_accuracy_dist = []
    cheaters_did_round = 0
    users_did_round = 0

    for page_id in range(pages_n):
        trust_judgment = 0
        while trust_judgment != judgment_min:
            w_page_judgment = {}
            worker_trust, worker_accuracy = pick_worker(user_prop, user_population)
            if worker_accuracy == 0.5:
                is_rand_ch = True
            else:
                is_rand_ch = False

            for row_id in range(page_id*rows_page, page_id*rows_page+rows_page, 1):
                gold_value = gold_data[row_id][0]
                # if the paper is average
                # OR
                # if the worker is a random cheater
                if gold_data[row_id][1] == 0 or is_rand_ch:
                    worker_judgment = np.random.binomial(1, worker_accuracy
                                                         if gold_value == 1
                                                         else 1 - worker_accuracy)
                else:
                    if worker_accuracy + easy_add_acc > 1.:
                        worker_accuracy_new = 1.
                    else:
                        worker_accuracy_new = worker_accuracy + easy_add_acc
                    worker_judgment = np.random.binomial(1, worker_accuracy_new
                                                         if gold_value == 1
                                                         else 1 - worker_accuracy_new)
                w_page_judgment.update({row_id: worker_judgment})
            new_worker_trust = get_trust(w_page_judgment, gold_data, test_page, worker_trust, quiz_papers_n)
            # is a worker passed tests rows
            if new_worker_trust >= trust_min:
                if is_rand_ch:
                    cheaters_did_round += 1
                    users_did_round += 1
                else:
                    users_did_round += 1
                worker_accuracy_dist.append(worker_accuracy)
                # add data to trusted_workers_judgment
                for row_id in w_page_judgment.keys():
                    trusted_judgment[row_id].append(w_page_judgment[row_id])
                trusted_workers_n += 1
                trust_judgment += 1
            else:
                untrusted_workers_n += 1

            # monetary issue
            budget_spent += price_page

    users_did_round_prop = [float(cheaters_did_round)/users_did_round,
                            float(users_did_round-cheaters_did_round)/users_did_round]
    paid_pages_n = trusted_workers_n+untrusted_workers_n
    return [trusted_judgment, budget_spent, paid_pages_n, worker_accuracy_dist, users_did_round_prop]


def do_task_scope(trust_min, test_page, papers_page, n_papers, price_row, judgment_min,
                  user_prop, user_population, easy_add_acc, quiz_papers_n, fp_cost, fn_cost):
    # generate gold data
    # [paper_x] = [[gold_val], [is_easy]]
    pages_n = n_papers / papers_page
    rows_page = test_page + papers_page
    total_papers_n = rows_page * pages_n
    gold_data = [(random.randint(0, 1), random.randint(0, 1)) for _ in range(total_papers_n)]
    round_res = do_round(trust_min, test_page, papers_page, n_papers, price_row, gold_data,
                         judgment_min, user_prop, user_population, easy_add_acc, quiz_papers_n)
    trusted_judgment = round_res[0]
    budget_spent = round_res[1]
    paid_pages_n = round_res[2]
    worker_accuracy_dist = round_res[3]
    users_did_round_prop = round_res[4]

    # delete test items before estimating the metrics
    add_val = test_page + papers_page
    tests_ids = range(test_page)
    for _ in range(pages_n-1):
        tests_ids += map(lambda x: x + add_val, tests_ids[-test_page:])
    for test_id in sorted(tests_ids, reverse=True):
        del gold_data[test_id]
        del trusted_judgment[test_id]

    acc, fp, fn, fp_lose, fn_lose = get_metrics(gold_data, trusted_judgment, fp_cost, fn_cost)
    return [budget_spent, paid_pages_n, worker_accuracy_dist,
            users_did_round_prop, acc, fp, fn, fp_lose, fn_lose]
