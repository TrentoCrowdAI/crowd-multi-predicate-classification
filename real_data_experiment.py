import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import copy
# import random
# import math
from m_run import m_run
from baseline import baseline
from sm_run import sm_run


J = 5
n_criteria = 3
n_papers = 100


def get_data():
    data = pd.read_csv('output/amt_data/crowd_data-2.csv')
    workers_ids_ch = set(list(pd.unique(data['intervention worker ID'])) \
                                   + list(pd.unique(data['use of tech worker ID'])) \
                                   + list(pd.unique(data['older adult worker ID'])))
    # workers_ids_ch = set(list(pd.unique(data['use of tech worker ID'])) \
    #                      + list(pd.unique(data['older adult worker ID'])))
    workers_ids_ch = [id_ch for id_ch in workers_ids_ch if type(id_ch) != float]
    workers_data = [[] for _ in workers_ids_ch]

    paper_ids_dict = dict(zip(set(data['paper ID']), range(n_papers)))
    # property of the data file
    criteria = {0: 'intervention Vote',
                1: 'use of tech vote',
                2: 'older adult vote'}
    column_workers_ids = ['intervention worker ID', 'use of tech worker ID', 'older adult worker ID']
    # criteria = {
    #             0: 'use of tech vote',
    #             1: 'older adult vote'}
    # column_workers_ids = ['use of tech worker ID', 'older adult worker ID']
    for w_id, w_ch in enumerate(workers_ids_ch):
        for c_id, c_name in criteria.iteritems():
            column_workers = column_workers_ids[c_id]
            c_data = data[data[column_workers] == w_ch][['paper ID', column_workers, c_name]]
            if not len(c_data):
                continue
            for row_index, row in c_data.iterrows():
                paper_id = paper_ids_dict[row['paper ID']]
                if row[c_name] == -1 or row[c_name] == 0:
                    vote = 1
                elif row[c_name] == 1:
                    vote = 0
                workers_data[w_id].append((paper_id, c_id, vote))
    # parse gold data
    GT = [None]*n_papers*n_criteria
    for paper_id_raw in paper_ids_dict.keys():
        paper_id = paper_ids_dict[paper_id_raw]
        for cr_id, cr_gold_colmn in zip(range(n_criteria), ['GOLD INTERVENTION', 'GOLD USE OF TECH', 'GOLD OLD']):
        # for cr_id, cr_gold_colmn in zip(range(n_criteria), ['GOLD USE OF TECH', 'GOLD OLD']):
            gold_value_raw = data[data['paper ID'] == paper_id_raw][cr_gold_colmn].values[0]
            gold_value = 0 if gold_value_raw == 1 else 1
            GT[paper_id * n_criteria + cr_id] = gold_value
    return workers_data, GT


def do_quiz(data, GT, Nt):
    workers_data = []
    for w in data:
        if len(w) >= Nt+5:
            tests_correct = 0
            # uiz_data = random.sample(w, Nt)
            # for v_data in quiz_data:
            for v_data in w[:Nt]:
                if v_data[2] == GT[v_data[0]*n_criteria + v_data[1]]:
                    tests_correct += 1
            if tests_correct == Nt:
                workers_data.append(w[Nt:])
    return workers_data


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def get_accuracy(c_votes, GT):
    criteria_accuracy = []
    for c in range(n_criteria):
        votes = []
        c_responses = c_votes[c::n_criteria]
        c_gt = GT[c::n_criteria]
        for p_c_votes, gt in zip(c_responses, c_gt):
            for vote in p_c_votes:
                if vote[1] == gt:
                    votes.append(1)
                else:
                    votes.append(0)
        criteria_accuracy.append(mean_confidence_interval(votes, confidence=0.95))
    return criteria_accuracy


def j_correction(c_votes, criteria_accuracy, GT, J):
    w_id = 3333
    c_votes_new = []
    counter = 0
    for index, votes in enumerate(c_votes):
        if len(votes) > J:
            c_votes_new.append(votes[:5])
        else:
            votes = votes[:]
            for _ in range(J - len(votes)):
                c_id  = index % n_criteria
                gt = GT[index]
                mean = criteria_accuracy[c_id][0]
                sigma = (criteria_accuracy[c_id][2] - criteria_accuracy[c_id][0]) / 2
                acc = np.random.normal(mean, sigma, 1)[0]
                if acc > 1.: acc = 1.
                v = np.random.binomial(gt, acc, 1)[0]
                votes.append((w_id, v))
                counter += 1
        c_votes_new.append(votes)
    return c_votes_new, counter


if __name__ == '__main__':
    lr = 5
    J = 5
    fr_p_part = 0.25
    for Nt in [1, 2, 4, 5]:
        print 'Nt: {}, J: {}'.format(Nt, J)
        w_data, GT = get_data()
        w_data = do_quiz(w_data, GT, Nt)
        c_votes = [[] for _ in range(n_criteria * n_papers)]
        for worker_id, worker_votes in enumerate(w_data):
            for paper_id, c_id, vote in worker_votes:
                c_votes[paper_id * n_criteria + c_id].append((worker_id, vote))
        criteria_accuracy = get_accuracy(c_votes, GT)

        loss_b, rec_b, pre_b, f_b, price_b = [], [], [], [], []
        loss_m, rec_m, pre_m, f_m, price_m = [], [], [], [], []
        loss_sm, rec_sm, pre_sm, f_sm, price_sm, syn_prop_sm = [], [], [], [], [], []

        for _ in range(10):
            cj_votes, counter = j_correction(c_votes, criteria_accuracy, GT, J)
            syn_votes_prop = float(counter) / (len(c_votes) * J)
            # Baseline
            loss_b_, rec_b_, pre_b_, f_beta_b, price_b_ = baseline(copy.deepcopy(cj_votes), n_criteria, n_papers, lr, GT)
            loss_b.append(loss_b_)
            rec_b.append(rec_b_)
            pre_b.append(pre_b_)
            f_b.append(f_beta_b)
            price_b.append(price_b_)

            # M-runs
            loss_m_, rec_m_, pre_m_, f_beta_m, price_m_ = m_run(copy.deepcopy(cj_votes), n_criteria, n_papers, lr, GT, fr_p_part)
            loss_m.append(loss_m_)
            rec_m.append(rec_m_)
            pre_m.append(pre_m_)
            f_m.append(f_beta_m)
            price_m.append(price_m_)

            # SM-runs
            loss_sm_, rec_sm_, pre_sm_, f_beta_sm, price_sm_, syn_prop_sm_ = sm_run(copy.deepcopy(cj_votes), n_criteria, n_papers,
                                                                               lr, GT, fr_p_part, criteria_accuracy)
            loss_sm.append(loss_sm_)
            rec_sm.append(rec_sm_)
            pre_sm.append(pre_sm_)
            f_sm.append(f_beta_sm)
            price_sm.append(price_sm_)
            syn_prop_sm.append(syn_prop_sm_)

        print 'BASELINE syn_votes_prop: {}, loss: {:1.2f}, price: {:1.2f}, recall: {:1.2f}, precision: {:1.2f}, f_b: {}'. \
            format(syn_votes_prop, np.mean(loss_b), np.mean(price_b), np.mean(rec_b), np.mean(pre_b), np.mean(f_b))

        print 'M-RUN    syn_votes_prop: {}, loss: {:1.2f}, price: {:1.2f}, recall: {:1.2f}, precision: {:1.2f}, f_b: {}'. \
            format(syn_votes_prop, np.mean(loss_m), np.mean(price_m), np.mean(rec_m), np.mean(pre_m), np.mean(f_m))

        print 'SM-RUN    syn_votes_prop: {}, loss: {:1.2f}, price: {:1.2f}, recall: {:1.2f}, precision: {:1.2f}, f_b: {}'. \
            format(np.mean(syn_prop_sm), np.mean(loss_sm), np.mean(price_sm), np.mean(rec_sm), np.mean(pre_sm), np.mean(f_sm))

        print '---------------------'

