'''
run experiments
'''

import numpy as np
import pandas as pd
from cf_simulation_synthetic import synthesizer
from quiz_simulation import do_quiz_scope
from task_simulation import do_task_scope, get_metrics
from generator import synthesize
from classifier_utils import classifier, estimate_accuracy, find_jt


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
    return [user_prop, user_population, acc_passed_distr]


def run_task_scope(trust_trsh, user_prop, user_population, easy_add_acc, n_papers, quiz_papers_n):
    tests_page = 0
    papers_page = 5
    price_row = 0.2
    fp_cost = 3
    data = []
    # do simulation
    for N in range(3, 21, 2):
        Nj_params = range(N/2 + 1, N + 1, 1)
        loss_dict = {}
        for key in Nj_params:
            loss_dict.update({key: []})  # {Nj: [loss]}
        for _ in range(1000):
            gold_data, trusted_judgment, budget_spent = do_task_scope(trust_trsh, tests_page, papers_page,
                                                                      n_papers, price_row, N, user_prop,
                                                                      user_population, easy_add_acc, quiz_papers_n)
            for Nj in Nj_params:
                loss = get_metrics(gold_data, trusted_judgment, fp_cost, Nj)
                loss_dict[Nj].append(loss)

        loss_avg_list, loss_std_list = [], []
        for Nj in Nj_params:
            loss_avg_list.append(np.average(loss_dict[Nj]))
            loss_std_list.append(np.std(loss_dict[Nj]))
        Nj_ind_opt = loss_avg_list.index(min(loss_avg_list))
        Nj_opt = Nj_params[Nj_ind_opt]
        data_iem = [quiz_papers_n, N, Nj_opt, loss_avg_list[Nj_ind_opt],
                    loss_std_list[Nj_ind_opt], budget_spent]
        data.append(data_iem)

    df = pd.DataFrame(data=data, columns=['tests_page', 'N', 'Nj', 'loss_avg',
                                          'loss_std', 'budget'])

    with open('visualisation/data/task_main_plot_in05.csv', 'a') as f:
        df.to_csv(f, header=False, index=False)


def run_task_criteria():
    tests_page_params = [1, 1, 1, 2, 2, 3]
    papers_page_params = [1, 2, 3, 2, 3, 3]
    for test_page, papaers_page in zip(tests_page_params, papers_page_params):
        job_accuracy_list = []
        budget_spent_list = []
        for _ in range(10000):
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


def postProc_algorithm():
    trusts_trsh = 0.
    cheaters_prop = 0.9
    n_papers = 600
    quiz_papers_n = 5
    papers_page = 5
    J = 5
    theta = 0.5
    cost = 3
    print 'theta: {}'.format(theta)

    user_prop, user_population, acc_distribution = run_quiz_scope(trusts_trsh, quiz_papers_n, cheaters_prop, 0.0)
    GT, psi_obj, psi_w = synthesize(acc_distribution, n_papers, papers_page, J, theta)
    Jt = J / 2 + 1
    print '-------------'
    for _ in range(5):
        print 'Jt: {}'.format(Jt)
        agg_values, theta_est = classifier(psi_obj, Jt)
        acc_avg = estimate_accuracy(agg_values, psi_w)
        Jt = find_jt(theta_est, J, acc_avg, cost)
        loss_real = get_metrics(GT, psi_obj, cost, Jt)
        print 'loss_real: {}'.format(loss_real)
        print 'theta_est: {}'.format(theta_est)
        print '-------'


if __name__ == '__main__':
    postProc_algorithm()
    # trusts_trsh = 1.
    # cheaters_prop = 0.3
    # easy_add_acc = 0.0
    # n_papers = 300

    # for quiz_papers_n in range(1, 11, 1):
    #     quiz_papers_n = 5
    #     user_prop, user_population, acc_distr = run_quiz_scope(trusts_trsh, quiz_papers_n, cheaters_prop, easy_add_acc)
    #     d_item = run_task_scope(trusts_trsh, user_prop, user_population, easy_add_acc, n_papers, quiz_papers_n)
