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


# def get_accuracy(gold_data, trusted_workers_judgment):
#     total_criteria_judgments = 0.
#     correct_criteria_judgments = 0.
#     for row_id, row_gold in enumerate(gold_data):
#         for row_judgment in trusted_workers_judgment[row_id]:
#             for gold, judg in zip(row_gold, row_judgment):
#                 if gold == judg:
#                     correct_criteria_judgments += 1
#                 total_criteria_judgments += 1
#     job_accuracy = correct_criteria_judgments/total_criteria_judgments
#     return job_accuracy


def pick_worker(user_prop, user_population):
    smart_ch = user_prop[1]
    t_worker = user_prop[2]
    p_val = np.random.uniform(0.0, 1.)
    if p_val < t_worker:
        worker_trust, worker_accuracy = user_population['worker'].pop()
    elif p_val >= t_worker and p_val < t_worker + smart_ch:
        worker_trust, worker_accuracy = user_population['smart_ch'].pop()
    else:
        worker_trust, worker_accuracy = user_population['rand_ch'].pop()
    return (worker_trust, worker_accuracy)


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
    return (trusted_judgment, budget_spent, paid_pages_n, worker_accuracy_dist, users_did_round_prop)


def do_task_scope(trust_min, test_page, papers_page, n_papers, price_row, judgment_min,
                  user_prop, user_population, easy_add_acc, quiz_papers_n):
    # generate gold data
    # [paper_x] = [[gold_val], [is_easy]]
    pages_n = n_papers / papers_page
    tests_n = test_page * pages_n
    total_papers_n = tests_n + n_papers
    gold_data = [(random.randint(0, 1), random.randint(0, 1)) for _ in range(total_papers_n)]
    round_res = do_round(trust_min, test_page, papers_page, n_papers, price_row, gold_data,
                         judgment_min, user_prop, user_population, easy_add_acc, quiz_papers_n)
    trusted_judgment = round_res[0]
    budget_spent = round_res[1]
    paid_pages_n = round_res[2]
    worker_accuracy_dist = round_res[3]
    users_did_round_prop = round_res[4]

    # job_accuracy = get_accuracy(gold_data, trusted_workers_judgment)
    # return (job_accuracy, budget_spent, paid_pages_n)
