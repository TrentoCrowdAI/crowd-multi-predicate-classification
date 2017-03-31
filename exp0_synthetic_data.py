import numpy as np


def generator(n_users, n_papers, acc_min, n_excl):
    '''
    Ground truth generation

    30% of papers are IN
    50% of papers are OUT
    20% of papers are MAYBE
    '''
    n_papers_in = int(0.3 * n_papers)
    n_papers_maybe = int(0.2 * n_papers)
    n_papers_out = n_papers - n_papers_in - n_papers_maybe

    ground_truth = [None] * n_papers_in + ['MAYBE'] * n_papers_maybe
    for paper in range(n_papers_out):
        gt_criteria = list(np.random.choice(range(n_excl), np.random.randint(2, n_excl), False))
        ground_truth.append(gt_criteria)

    # Answers generation
    users_accracy = np.random.uniform(acc_min, .7, n_users)
    answers = []
    for paper_id in range(n_papers):
        truth = ground_truth[paper_id]
        answ_for_paper = []
        for user_id in range(n_users):
            user_acc = users_accracy[user_id]
            if np.random.binomial(1, user_acc):
                # assign true value
                # if true value is "OUT"
                if isinstance(truth, list):
                    criteria_from_user = []
                    for i in truth:
                        if np.random.binomial(1, user_acc):
                            criteria_from_user.append(i)
                    if len(criteria_from_user) == 0:
                        criteria_from_user.append(truth[:2])
                    answ_for_paper.append(criteria_from_user)
                else:
                    answ_for_paper.append(truth)
            else:
                # assign false value
                # TO DO: 'F criteria'
                if isinstance(truth, list):
                    if np.random.binomial(1, 0.5):
                        answ_for_paper.append('Maybe')
                    else:
                        answ_for_paper.append(None)
                elif truth:
                    if np.random.binomial(1, 0.5):
                        answ_for_paper.append(None)
                    else:
                        answ_for_paper.append(['F criteria'])
                else:
                    if np.random.binomial(1, 0.5):
                        answ_for_paper.append('Maybe')
                    else:
                        answ_for_paper.append(['F criteria'])
        answers.append(answ_for_paper)

    return answers, ground_truth


def majority_voting(answers):
    results = []
    for paper_answ in answers:
        frequencies = {None: 0,
                       'MAYBE': 0,
                       'OUT': 0}
        for answ in paper_answ:
            if isinstance(answ, list):
                frequencies['OUT'] += 1
            elif answ:
                frequencies['MAYBE'] += 1
            else:
                frequencies[None] += 1

        mv = max(frequencies, key=frequencies.get)
        if mv == 'OUT':
            mv = ['F criteria']
        results.append(mv)

    return results


def get_accuracy(data, ground_truth):
    count = 0.
    for mv, gt in zip(data, ground_truth):
        if mv == gt:
            count += 1
        elif type(mv) == type(gt):
            count += 1

    return count/len(data)


if __name__ == '__main__':

    # experiment
    # res_list = [range(3, 30, 1), [], []]
    # for n_users in range(3, 30, 1):
    #     temp_res = []
    #     for i in range(100):
    #         answers, ground_truth = generator(n_users=n_users, n_papers=10, acc_min=0.5, n_excl=8)
    #         mv_result = majority_voting(answers=answers)
    #         temp_res.append(get_accuracy(mv_result, ground_truth))
    #     res_list[1].append(np.mean(temp_res))
    #     res_list[2].append(np.std(temp_res))


    # print res_lis
    # t
    temp_res = []
    for i in range(100):
            answers, ground_truth = generator(n_users=3, n_papers=5, acc_min=0.5, n_excl=8)
            mv_result = majority_voting(answers=answers)
            temp_res.append(get_accuracy(mv_result, ground_truth))
    print np.mean(temp_res)
    print np.std(temp_res)
