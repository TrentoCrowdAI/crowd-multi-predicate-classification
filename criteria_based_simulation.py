from quiz_simulation import do_quiz_criteria_confm
from generator import generate_responses_gt
from fusion_algorithms.dawid_skene import dawid_skene


def run_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty):
    acc_passed_distr = [[], []]
    for _ in range(100000):
        result = do_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty)
        if len(result) > 1:
            acc_passed_distr[0].append(result[0])
            acc_passed_distr[1].append(result[1])
    return acc_passed_distr


if __name__ == '__main__':
    z = 0.3
    cost = 5
    n_papers = 1000
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    for Nt in range(1, 11, 1):
        acc = run_quiz_criteria_confm(Nt, z, criteria_difficulty)
        for J in [2, 3, 5, 10]:
            responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page, J, acc, criteria_difficulty)
            values_prob = dawid_skene(responses, tol=0.00001, max_iter=10)  # TO DO max_iter=100
            # - post processing
            # - get metrics
            pass
