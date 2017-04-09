'''
run experiments
'''

import numpy as np
from cf_simulation_synthetic import synthesizer
from quiz_simulation import do_quiz_scope
from task_simulation import do_task_scope


def run_quiz_scope(trust_min=0.75, quiz_papers_n=4, cheaters_prop=0.5):
    statistic_passed = {
        'rand_ch': 0,
        'smart_ch': 0,
        'worker': 0
    }
    statistic_total = {
        'rand_ch': 0,
        'smart_ch': 0,
        'worker': 0
    }
    for _ in range(1000):
        result = do_quiz_scope(trust_min, quiz_papers_n, cheaters_prop)
        if len(result) > 1:
            statistic_passed[result[2]] += 1
            statistic_total[result[2]] += 1
        else:
            statistic_total[result[0]] += 1

    rand_cheaters_passed = statistic_passed['rand_ch'] / float(statistic_total['rand_ch']) * 100
    smart_cheaters_passed = statistic_passed['smart_ch'] / float(statistic_total['smart_ch']) * 100
    workers_passed = statistic_passed['worker'] / float(statistic_total['worker']) * 100

    print 'random cheaters passed: {}%'.format(rand_cheaters_passed)
    print 'smart cheaters passed: {}%'.format(smart_cheaters_passed)
    print 'workers passed: {}%'.format(workers_passed)

#   calculate the proportion of types of users passed the quiz
    user_prop = []
    users_passed = float(sum(statistic_passed.values()))
    for user_t in ['rand_ch', 'smart_ch', 'worker']:
        user_prop.append(statistic_passed[user_t]/users_passed)
    return user_prop


def run_task_scope(trust_min, user_prop):
    # params for the task_scope function
    tests_page_params = [1, 1, 1, 2, 2, 3]
    papers_page_params = [1, 2, 3, 2, 3, 3]
    for test_page, papaers_page in zip(tests_page_params, papers_page_params):
        job_accuracy_list = []
        budget_spent_list = []
        for _ in range(1000):
            do_task_scope()


def run_task_criteria():
    tests_page_params = [1, 1, 1, 2, 2, 3]
    papers_page_params = [1, 2, 3, 2, 3, 3]
    for test_page, papaers_page in zip(tests_page_params, papers_page_params):
        job_accuracy_list = []
        budget_spent_list = []
        for _ in range(1000):
            job_accuracy, budget_spent, paid_pages_n = synthesizer(trust_min=1., n_criteria=3,
                                                                   test_page=test_page, papers_page=papaers_page,
                                                                   quiz_papers_n=4, n_papers=18, budget=50,
                                                                   price_row=0.4, judgment_min=3, judgment_max=5,
                                                                   cheaters_prop=0.1)
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


if __name__ == '__main__':
    trust_min = 0.75
    quiz_papers_n = 4
    cheaters_prop = 0.5
    user_prop = run_quiz_scope(trust_min, quiz_papers_n, cheaters_prop)
    run_task_scope(trust_min, user_prop)

