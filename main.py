'''
run experiments
'''

from cf_simulation_synthetic import synthesizer
import numpy as np

if __name__ == '__main__':
    tests_page_params = [1, 1, 1, 2, 2, 3]
    papers_page_params = [1, 2, 3, 2, 3, 3]
    for test_page, papaers_page in zip(tests_page_params, papers_page_params):
        job_accuracy_list = []
        budget_spent_list = []
        for _ in range(1000):
            job_accuracy, budget_spent, paid_pages_n = synthesizer(trust_min=1., n_criteria=3,
                        test_page=test_page, papers_page=papaers_page, quiz_papers_n=4, n_papers=18, budget=50,
                        price_row=0.4, judgment_min=3, judgment_max=5, cheaters_prop=0.1)
            job_accuracy_list.append(job_accuracy)
            budget_spent_list.append(budget_spent)

        job_accuracy_avg = np.mean(job_accuracy_list)
        job_accuracy_std = np.std(job_accuracy_list)
        budget_spent_avg = np.mean(budget_spent_list)
        budget_spent_std = np.std(budget_spent_list)

        print '*********************'
        print 'tests_page: {}'.format(test_page)
        print 'papaers_page: {}'.format(papaers_page)
        print '---------------------'
        print 'job_accuracy_avg={}\n' \
              'job_accuracy_std={}\n' \
              'budget_spent_avg={}$\n' \
              'budget_spent_std={}$\n'.format(job_accuracy_avg, job_accuracy_std, budget_spent_avg, budget_spent_std)
