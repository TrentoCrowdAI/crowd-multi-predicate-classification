'''
run experiments
'''

from cf_simulation_synthetic import synthesizer

if __name__ == '__main__':
    job_accuracy, budget_spent, paid_pages_n = synthesizer(trust_min=0.75, n_criteria=3,
                test_page=1, papers_page=3, quiz_papers_n=4, n_papers=18, budget=50,
                price_row=0.4, judgment_min=3, judgment_max=5, cheaters_prop=0.1)

    print 'job_accuracy={}\nbudget_spent={}$\npaid_pages_n={}\n'.format(job_accuracy, budget_spent, paid_pages_n)
