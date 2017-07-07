import numpy as np

def assign_best_criteria():
    pass


def filter_papers():
    pass


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
    return classified_papers, classified_papers_ids, rest_papers_ids


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
