def assign_best_criteria():
    pass


def filter_papers():
    pass


def classify_papers(papers_ids, criteria_num, values_prob, lr):
    classified_papers = []
    classified_papers_ids = []
    rest_papers_ids = []

    trsh = lr / (lr + 1.)
    for paper_id in papers_ids:
        p_inclusion = 1.
        for e_paper_id in range(criteria_num):
            p_inclusion *= values_prob[paper_id * criteria_num + e_paper_id][0]
        p_exclusion = 1 - p_inclusion

        if p_exclusion > trsh:
            classified_papers.append(0)
            classified_papers_ids.append(paper_id)
        elif p_inclusion > trsh:
            classified_papers.append(1)
            classified_papers_ids.append(paper_id)
        else:
            rest_papers_ids.append(paper_id)
    return classified_papers, classified_papers_ids, rest_papers_ids
