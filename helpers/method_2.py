import numpy as np


def assign_criteria(papers_ids, criteria_num, values_prob, acc_cr_list):
    cr_assigned = []
    cr_list = range(criteria_num)
    for p_id in papers_ids:
        p_classify = []
        for cr in cr_list:
            prior_out = values_prob[p_id * criteria_num + cr][1]
            acc_cr = acc_cr_list[cr]
            p_classify.append(prior_out * acc_cr)
        cr_assign = p_classify.index(max(p_classify))
        cr_assigned.append(cr_assign)
    return cr_assigned


def classify_papers(papers_ids, criteria_num, values_prob, lr):
    classified_papers = []
    classified_papers_ids = []
    rest_papers_ids = []
    trsh = lr / (lr + 1.)
    for paper_id in papers_ids:
        p_inclusion = 1.
        for e_paper_id in range(criteria_num):
            p_inclusion *= values_prob[paper_id * criteria_num + e_paper_id][0]
        p_exclusion = 1 - p_inclusion

        if p_exclusion > trsh:
            classified_papers.append(0)
            classified_papers_ids.append(paper_id)
        elif p_inclusion > trsh:
            classified_papers.append(1)
            classified_papers_ids.append(paper_id)
        else:
            rest_papers_ids.append(paper_id)
    return dict(zip(classified_papers_ids, classified_papers)), rest_papers_ids


def generate_responses(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
    responses = []
    n = len(papers_ids)
    workers_n = 1 if n < papers_worker else n / papers_worker
    for w_ind in range(workers_n):
        worker_acc_in = acc[1].pop()
        acc[1].insert(0, worker_acc_in)
        worker_acc_out = acc[0].pop()
        acc[0].insert(0, worker_acc_out)
        for cr, p_id in zip(cr_assigned[w_ind*papers_worker: w_ind*papers_worker+papers_worker],
                            papers_ids[w_ind*papers_worker: w_ind*papers_worker+papers_worker]):
            cr_vals_id = range(p_id * criteria_num, p_id * criteria_num + criteria_num, 1)
            isPaperIN = sum([GT[i] for i in cr_vals_id]) == 0
            if isPaperIN:
                worker_acc = worker_acc_in
            else:
                worker_acc = worker_acc_out

            GT_cr = GT[p_id * criteria_num + cr]
            cr_dif = criteria_difficulty[cr]
            if np.random.binomial(1, worker_acc * cr_dif if worker_acc * cr_dif <= 1. else 1.):
                vote = GT_cr
            else:
                vote = 1 - GT_cr
            responses.append(vote)
    return responses


def update_v_prob(values_prob, responses, p_ids, cr_assigned, criteria_num, acc_cr_list):
    for cr, vote, p_id in zip(cr_assigned, responses, p_ids):
        p_prob = values_prob[p_id*criteria_num+cr]
        acc_cr = acc_cr_list[cr]
        prop_cr_in = p_prob[0]*(1-acc_cr) if vote else p_prob[0]*acc_cr
        prop_cr_out = p_prob[1]*acc_cr if vote else p_prob[1]*(1-acc_cr)
        norm_const = prop_cr_in + prop_cr_out
        prob_cr_in = prop_cr_in / norm_const
        prob_cr_out = prop_cr_out / norm_const
        values_prob[p_id * criteria_num + cr][0] = prob_cr_in
        values_prob[p_id * criteria_num + cr][1] = prob_cr_out
