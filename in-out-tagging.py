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
    users_accracy = np.random.uniform(acc_min, 1., n_users)
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


if __name__ == '__main__':
    answers, ground_truth = generator(n_users=5, n_papers=10, acc_min=0.5, n_excl=8)
    majority_voting(answers=answers)
