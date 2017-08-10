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
workers_data = [[] for _ in workers_ids_ch]
c_votes = [[] for _ in range(n_criteria * n_papers)]

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
            paper_id = row['paper ID']
            vote = row[c_name]
            workers_data[w_id].append((paper_id, c_id, vote))
            # c_votes[paper_id * n_criteria + c_id] =
pass




# data_summary = pd.DataFrame(data=data_summary_temp, columns=['n_correct', 'n_total'])
# data_summary = data_summary.groupby(['n_correct', 'n_total']).size().reset_index(name="Time")

# for _, row in data.iterrows():
#     worker_id = int(row['worker_id'])
