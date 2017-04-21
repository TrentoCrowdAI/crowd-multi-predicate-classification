'''
run experiments
'''

import numpy as np
import pandas as pd
from cf_simulation_synthetic import synthesizer
from quiz_simulation import do_quiz_scope
from task_simulation import do_task_scope


def run_quiz_scope(trust_min=0.75, quiz_papers_n=4, cheaters_prop=0.5,  easy_add_acc = 0.2):
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
    # a value: (user_trust, user_accuracy)
    user_population = {
        'rand_ch': [],
        'smart_ch': [],
        'worker': []
    }
    acc_passed_distr = []
    for _ in range(10000):
        result = do_quiz_scope(trust_min, quiz_papers_n, cheaters_prop, easy_add_acc)
        if len(result) > 1:
            statistic_passed[result[2]] += 1
            statistic_total[result[2]] += 1
            user_population[result[2]].append(result[:2])
            acc_passed_distr.append(result[1])
        else:
            statistic_total[result[0]] += 1

    rand_cheaters_passed = statistic_passed['rand_ch'] / float(statistic_total['rand_ch']) * 100
    # smart_cheaters_passed = statistic_passed['smart_ch'] / float(statistic_total['smart_ch']) * 100
    smart_cheaters_passed = 0.
    workers_passed = statistic_passed['worker'] / float(statistic_total['worker']) * 100

    print '*** Quiz ***'
    print 'random cheaters passed: {}%'.format(rand_cheaters_passed)
    print 'smart cheaters passed: {}%'.format(smart_cheaters_passed)
    print 'workers passed: {}%'.format(workers_passed)

#   calculate the proportion of types of users passed the quiz
    user_prop = []
    users_passed = float(sum(statistic_passed.values()))
    for user_t in ['rand_ch', 'smart_ch', 'worker']:
        user_prop.append(statistic_passed[user_t]/users_passed)
    return [user_prop, user_population]


def run_task_scope(trust_min, user_prop, user_population, easy_add_acc, quiz_papers_n, n_papers):
    # params for the do_task_scope function
    # tests_page_params = [1, 1, 1, 2, 2, 3]
    # papers_page_params = [1, 2, 3, 2, 3, 3]
    tests_page_params = [1]*9
    papers_page_params = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    price_row = 0.2
    judgment_min = 3
    fp_cost = 3
    fn_cost = 1
    data = []
    accuracy_data = []
    # do simulation
    for test_page, papers_page in zip(tests_page_params, papers_page_params):
        for judgment_min in [3, 5, 7]:
            # for statistics
            budget_spent_list = []
            paid_pages_n_list = []
            worker_accuracy_dist = []
            users_did_round_prop_list = [[], []]
            acc_list = []
            fp_list = []
            fn_list = []
            fp_lose_list = []
            fn_lose_list = []
            for _ in range(1000):
                task_results = do_task_scope(trust_min, test_page, papers_page, n_papers, price_row, judgment_min,
                                             user_prop, user_population, easy_add_acc, quiz_papers_n, fp_cost, fn_cost)
                budget_spent_list.append(task_results[0])
                paid_pages_n_list.append(task_results[1])
                worker_accuracy_dist += task_results[2]
                users_did_round_prop_list[0].append(task_results[3][0])
                users_did_round_prop_list[1].append(task_results[3][1])
                acc_list.append(task_results[4])
                fp_list.append(task_results[5])
                fn_list.append(task_results[6])
                fp_lose_list.append(task_results[7])
                fn_lose_list.append(task_results[8])

            budget_spent_avg = np.average(budget_spent_list)
            paid_pages_n_avg = np.average(paid_pages_n_list)
            users_did_round_prop_avg = [np.average(users_did_round_prop_list[0]),
                                        np.average(users_did_round_prop_list[1])]
            acc_avg = np.average(acc_list)
            acc_std = np.std(acc_list)
            fp_avg = np.average(fp_list)
            fn_avg = np.average(fn_list)
            fp_lose_avg = np.average(fp_lose_list)
            fp_lose_std = np.std(fp_lose_list)
            fn_lose_avg = np.average(fn_lose_list)
            fn_lose_std = np.std(fn_lose_list)

            data.append([test_page, papers_page, trust_min, quiz_papers_n, n_papers, price_row, judgment_min,
                         fp_cost, fn_cost, budget_spent_avg, paid_pages_n_avg, users_did_round_prop_avg[0],
                         users_did_round_prop_avg[1], acc_avg, fp_avg, fn_avg, fp_lose_avg, fn_lose_avg,
                         acc_std, fp_lose_std, fn_lose_std])
            accuracy_data.append([test_page, papers_page, trust_min, quiz_papers_n, n_papers, price_row, judgment_min,
                             fp_cost, fn_cost]+acc_list)

            print '\n*** Task execution ***'
            print 'tests per page: {}'.format(test_page)
            print 'papers per page: {}'.format(papers_page)
            print '-----------------------'
            print 'budget_spent_avg: ${}'.format(budget_spent_avg)
            print 'paid_pages_n_avg: {}'.format(paid_pages_n_avg)
            print 'users_did_round_prop_avg: {}'.format(users_did_round_prop_avg)
            print 'acc_avg: {}'.format(acc_avg)
            print 'fp_avg: {}'.format(fp_avg)
            print 'fn_avg: {}'.format(fn_avg)
            print 'fp_lose_avg: {}'.format(fp_lose_avg)
            print 'fn_lose_avg: {}'.format(fn_lose_avg)

    res_frame = pd.DataFrame(data=data,
                             columns=['test_page', 'papers_page', 'trust_min', 'quiz_papers_n',
                                      'n_papers', 'price_row', 'judgment_min', 'fp_cost', 'fn_cost',
                                      'budget_spent_avg', 'paid_pages_n_avg', 'ch_did_round_prop',
                                      'wrk_did_round_prop', 'acc_avg', 'fp_avg', 'fn_avg', 'fp_lose_avg',
                                      'fn_lose_avg', 'acc_std', 'fp_lose_std', 'fn_lose_std'])
    # res_frame.to_csv('visualisation/task_stat_{}_{}.csv'.format(trust_min, cheaters_prop), index=False)
    res_frame.to_csv('visualisation/task_results_plot2_test.csv'.format(trust_min, cheaters_prop), index=False)

    # accuracy_columns = ['test_page', 'papers_page', 'trust_min', 'quiz_papers_n',
    #                     'n_papers', 'price_row', 'judgment_min', 'fp_cost', 'fn_cost'] \
    #                      + range(len(acc_list))
    # accuracy_frame = pd.DataFrame(data=accuracy_data, columns=accuracy_columns)
    # accuracy_frame.to_csv('visualisation/accuracy.csv', index=False)


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
    trust_min = 1.
    quiz_papers_n = 4
    cheaters_prop = 0.25
    easy_add_acc = 0.0
    n_papers = 300

    print '*** Set up ***'
    print 'quiz_papers_n: {}'.format(quiz_papers_n)
    print 'n_papers: {}'.format(n_papers)
    print 'trust_thrh: {}\n'.format(trust_min)

    user_prop, user_population = run_quiz_scope(trust_min, quiz_papers_n, cheaters_prop, easy_add_acc)
    run_task_scope(trust_min, user_prop, user_population, easy_add_acc, quiz_papers_n, n_papers)
