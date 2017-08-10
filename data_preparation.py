import math
import pandas as pd
import numpy as np

data = pd.read_csv('output/amt_data/table_1.csv')

workers_ids_ch = list(pd.unique(data['intervention worker ID']))\
                      + list(pd.unique(data['use of tech worker ID']))\
                      + list(pd.unique(data['older adult worker ID']))
# iterate over data rows
# compute statistic of answers
J = 5
n_criteria = 3
n_papers = len(data) / J
c_votes = [[] for _ in range(n_criteria * n_papers)]


def get_data():
    workers_data = [[] for _ in workers_ids_ch]
    paper_ids_dict = dict(zip(set(data['paper ID']), range(n_papers)))
    # property of the data file
    criteria = {0: 'intervention Vote',
                1: 'use of tech vote',
                2: 'older adult vote'}
    column_workers_ids = ['intervention worker ID', 'use of tech worker ID', 'older adult worker ID']
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
                else:
                    vote = 0
                workers_data[w_id].append((paper_id, c_id, vote))
    # parse gold data
    GT = [None]*n_papers*n_criteria
    for paper_id_raw in paper_ids_dict.keys():
        paper_id = paper_ids_dict[paper_id_raw]
        for cr_id, cr_gold_colmn in zip(range(3), ['GOLD INTERVENTION', 'GOLD USE OF TECH', 'GOLD OLD']):
            gold_value_raw = data[data['paper ID'] == paper_id_raw][cr_gold_colmn].values[0]
            gold_value = 0 if gold_value_raw == 1 else 1
            GT[paper_id * n_criteria + cr_id] = gold_value
    return workers_data, GT

    # Filter unqualified workers Stattistics
    # qualt_workers = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    # worker_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    # for Nt in [1, 2, 3, 4, 5]:
    #     for w in workers_data:
    #         if len(w) >= Nt+5:
    #             worker_counter[Nt] += 1
    #             tests_correct = 0
    #             for v_data in w[:Nt]:
    #                 if v_data[2] == GT[v_data[0]*n_criteria + v_data[1]]:
    #                     tests_correct += 1
    #             if tests_correct == Nt:
    #                 qualt_workers[Nt] += 1


if __name__ == '__main__':
    data, GT = get_data()
    pass

