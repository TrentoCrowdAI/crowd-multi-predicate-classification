from quiz_simulation import do_quiz_criteria_confm


def run_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty):
    acc_passed_distr = [[], []]
    for _ in range(100000):
        result = do_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty)
        if len(result) > 1:
            acc_passed_distr[0].append(result[0])
            acc_passed_distr[1].append(result[1])
    return acc_passed_distr


def classify_papers(n_papers, criteria_num, values_prob, cost):
    classified_papers = []
    exclusion_trsh = cost / (cost + 1.)
    for paper_id in range(n_papers):
        p_exclusion = 1.
        for e_paper_id in range(criteria_num):
            p_exclusion *= 1 - values_prob[paper_id*criteria_num+e_paper_id][1]
        p_exclusion = 1 - p_exclusion
        classified_papers.append(1) if p_exclusion > exclusion_trsh else classified_papers.append(0)
    return classified_papers
