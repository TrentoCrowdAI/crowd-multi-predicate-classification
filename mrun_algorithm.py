from fusion_algorithms.em import expectation_maximization
from fusion_algorithms.algorithms_utils import input_adapter
from utils import classify_papers, get_actual_loss
from generator import generate_responses_gt



# def estimate_cr_power_dif(responses, criteria_num, n_papers, papers_page, J):
#     Psi = input_adapter(responses)
#     N = (n_papers / papers_page) * J
#     p = expectation_maximization(N, n_papers * criteria_num, Psi)[1]
#     values_prob = []
#     for e in p:
#         e_prob = [0., 0.]
#         for e_id, e_p in e.iteritems():
#             e_prob[e_id] = e_p
#         values_prob.append(e_prob)
#
#     cr_difficulty = [0. for _ in range(criteria_num)]
#     cr_power = [0. for _ in range(criteria_num)]
#     for paper_id in range(len(responses)/criteria_num):
#         for e_id in range(criteria_num):
#             prob_apply_e = values_prob[paper_id*criteria_num+e_id][1]
#             cr_power[e_id] += prob_apply_e
#             cr_difficulty[e_id] += abs(2*prob_apply_e-1)
#
#     cr_dif_norm_const = sum(cr_difficulty)
#     cr_difficulty = map(lambda x: 1-x/cr_dif_norm_const, cr_difficulty)
#     cr_power_norm_const = sum(cr_power)
#     cr_power = map(lambda x: x/cr_power_norm_const, cr_power)
#     return cr_power, cr_difficulty

def first_round(responses, criteria_num, n_papers, papers_page, J, cost):
    classified_papers = classify_papers(n_papers, criteria_num, responses, papers_page, J, cost)

    # TO DO!!!
    best_cr_order = range(criteria_num)

    return classified_papers, best_cr_order


def do_round(GT, cr, papers_ids_rest, criteria_num, papers_worker, J,
             cost, acc, criteria_power, criteria_difficulty):
    n_papers = len(papers_ids_rest)
    GT_round = [GT[p_id*criteria_num+cr] for ind, p_id in enumerate(papers_ids_rest)]
    responses_round = generate_responses_gt(n_papers, [criteria_power[cr]], papers_worker,
                                            J, acc, [criteria_difficulty[cr]], GT_round)
    classified_papers = zip(papers_ids_rest, classify_papers(n_papers, 1, responses_round, papers_worker, J, cost))
    return classified_papers


def get_loss_cost_mrun(criteria_num, n_papers, papers_page, J, cost, Nt,
                       acc, criteria_power, criteria_difficulty, GT):
    # first round responses
    # 10% papers
    criteria_count = criteria_num * J * n_papers * 0.1
    GT_fround = GT[: int(n_papers*criteria_num*0.1)]
    responses_fround = generate_responses_gt(n_papers/10, criteria_power, papers_page,
                                             J, acc, criteria_difficulty, GT_fround)
    classified_papers_fround, best_cr_order = first_round(responses_fround, criteria_num,
                                                          n_papers/10, papers_page, J, cost)
    # Do Multi rounds
    papers_ids_rest = range(n_papers/10, n_papers, 1)
    classified_papers = classified_papers_fround + [1 for _ in papers_ids_rest]

    papers_worker = papers_page * criteria_num
    for cr in best_cr_order:
        n_rest = len(papers_ids_rest)
        criteria_count += J * n_rest
        papers_ids_rest1 = papers_ids_rest[:n_rest - n_rest % papers_worker]
        papers_ids_rest2 = papers_ids_rest[n_rest - n_rest % papers_worker:]
        classified_papers_cr = do_round(GT, cr, papers_ids_rest1, criteria_num, papers_worker, J,
                                        cost, acc, criteria_power, criteria_difficulty)
        # check if n_papers_rest % papers_page != 0 then run an additional round
        if papers_ids_rest2:
            classified_papers_cr += do_round(GT, cr, papers_ids_rest2, criteria_num, n_rest % papers_worker, J,
                                             cost, acc, criteria_power, criteria_difficulty)
        papers_ids_rest = []
        for p_id, p_cr in classified_papers_cr:
            if p_cr:
                papers_ids_rest.append(p_id)
            else:
                classified_papers[p_id] = 0
    loss = get_actual_loss(classified_papers, GT, cost, criteria_num)
    cost = criteria_count / float(n_papers)
    return loss, cost
