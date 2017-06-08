from quiz_simulation import do_quiz_criteria_confm


def run_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty):
    acc_passed_distr = [[], []]
    for _ in range(100000):
        result = do_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty)
        if len(result) > 1:
            acc_passed_distr[0].append(result[0])
            acc_passed_distr[1].append(result[1])
    return acc_passed_distr


def aggregate_papers(n_papers, criteria_num, values_prob):
    papers_prob_in = []
    for paper_id in range(n_papers):
        prob_in = 1.
        for e_paper_id in range(criteria_num):
            prob_in *= 1 - values_prob[paper_id*criteria_num+e_paper_id][1]
        papers_prob_in.append(prob_in)
    return papers_prob_in


def estimate_loss(papers_prob_in, cost):
    inclusion_trsh = 1 - cost / (cost + 1.)
    loss_total = 0.
    for p_in in papers_prob_in:
        if p_in > inclusion_trsh:
            loss_total += 1 - p_in
        else:
            loss_total += cost * p_in
    loss_per_paper = loss_total / len(papers_prob_in)
    return loss_per_paper


# def get_loss(classified_papers, GT, cost, criteria_num):
#     # obtain GT scope values for papers
#     GT_scope = []
#     for paper_id in range(len(classified_papers)):
#         if sum([GT[paper_id * criteria_num + e_paper_id] for e_paper_id in range(criteria_num)]):
#             GT_scope.append(0)
#         else:
#             GT_scope.append(1)
#     i_gt = sum(GT_scope)
#     e_gt = len(GT_scope) - i_gt
#     fe_num = 0.
#     fi_num = 0.
#     for cl_val, gt_val in zip(classified_papers, GT_scope):
#         if gt_val and not cl_val:
#             fe_num += 1
#         if not gt_val and cl_val:
#             fi_num += 1
#     loss = fe_num / i_gt * cost + fi_num / e

# def classify_papers(n_papers, criteria_num, values_prob, cost):
#     classified_papers = []
#     exclusion_trsh = cost / (cost + 1.)
#     for paper_id in range(n_papers):
#         p_exclusion = 1.
#         for e_paper_id in range(criteria_num):
#             p_exclusion *= 1 - values_prob[paper_id*criteria_num+e_paper_id][1]
#         p_exclusion = 1 - p_exclusion
#         classified_papers.append(1) if p_exclusion > exclusion_trsh else classified_papers.append(0)
#     return classified_papers
