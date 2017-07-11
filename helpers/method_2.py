import numpy as np
from scipy.special import binom


def assign_criteria(papers_ids, criteria_num, values_count, power_cr_list, acc_cr_list):
    cr_assigned = []
    cr_list = range(criteria_num)
    for p_id in papers_ids:
        p_classify = []
        for cr in cr_list:
            acc_cr = acc_cr_list[cr]
            power_cr = power_cr_list[cr]
            cr_count = values_count[p_id * criteria_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]

            # new value is out
            term1_p_out = binom(in_c+out_c+1, out_c+1)*acc_cr**(out_c+1)*(1-acc_cr)**in_c*power_cr
            term1_p_in = binom(in_c+out_c+1, in_c)*acc_cr**(in_c)*(1-acc_cr)**(out_c+1)*(1-power_cr)
            prob_pout_vout = term1_p_out/(term1_p_out+term1_p_in)

            # new value is in
            term2_p_out = binom(in_c+out_c+1, out_c)*acc_cr**out_c*(1-acc_cr)**(in_c+1)*power_cr
            term2_p_in = binom(in_c+out_c+1, in_c+1)*acc_cr**(in_c+1)*(1-acc_cr)**out_c*(1-power_cr)
            prob_pout_vin = term2_p_out/(term2_p_out+term2_p_in)

            p_out_cr = prob_pout_vout*power_cr + prob_pout_vin*(1-power_cr)
            p_classify.append(p_out_cr)
        cr_assign = p_classify.index(max(p_classify))
        cr_assigned.append(cr_assign)
    return cr_assigned


def classify_papers_baseline(papers_ids, criteria_num, values_prob, lr):
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


def update_v_count(values_count, criteria_num, cr_assigned, responses, p_ids):
    for cr, vote, p_id in zip(cr_assigned, responses, p_ids):
        if vote:
            values_count[p_id * criteria_num + cr][1] += 1
        else:
            values_count[p_id * criteria_num + cr][0] += 1
