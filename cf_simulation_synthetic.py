'''
This script models the workflow of crowdflower platform
 for the job of scope tagging scientific papers by crowd users.
The experiment uses synthetic data generated in the current script.

Parameters for the crowdflower synthesizer:
    trust_min - workers' trust threshold,
    n_criteria - number of exclusion criteria,
    test_pages - number of test questions per page,
    papers_page - number of papers(excluding test questions) per page,
    n_pages - number of pages to be tagged by a user,
    n_papers - set of papers to be tagged(excluding test questions), must: n_papers % papers_page = 0,
    budget - assumed total budget in dollars for the job; note: final budget may differs from initially assumed one,
    price_row - price per one row in dollars,
    judgment_min - min number judgments per row,
    judgment_max - max number judgments per row,
    cheaters_prop - proportion of cheaters from whole population of workers

    OUTPUT: accuracy of results, spent budget, total number of judgments being paid

'''

import random


def first_round():
    pass


def second_round():
    pass


def get_accuracy():
    pass


'''
gold_data = [[criteria_0 value, criteria_1 value,...],
             [criteria_0 value, criteria_1 value,...], [],..]

criteria_id value is a binary variable
each list in gold data presents criteria values for a paper with id=list's index in the gold_data
'''
def generate_gold_data(n_criteria=3, test_pages=1, papers_page=3, n_papers=30):
    gold_data = []
    pages_n = n_papers/papers_page
    tests_n = test_pages * pages_n
    total_papers_n = tests_n + n_papers
    for paper_id in range(total_papers_n):
        # decide if paper in/out of scope
        # bernoulli trial with p=0.5
        if random.randint(0, 1):
            paper_id_gold_data = [0]*n_criteria
        else:
            paper_id_gold_data = [random.randint(0, 1) for _ in range(n_criteria)]
        gold_data.append(paper_id_gold_data)
    return gold_data


def synthesizer(trust_min=0.75, n_criteria=3, test_pages=1, papers_page=3,
                n_pages=1, n_papers=30, budget=50, price_row=0.4,
                judgment_min=3, judgment_max=5, cheaters_prop=0.2):
    gold_data = generate_gold_data(n_criteria=3, test_pages=1, papers_page=3, n_papers=30)





if __name__ == '__main__':
    synthesizer()
