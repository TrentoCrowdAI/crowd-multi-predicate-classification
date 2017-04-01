'''
This script models the workflow of crowdflower platform
 for the job of scope tagging scientific papers by crowd users.
The experiment uses synthetic data generated in the current script.

Parameters for the crowdflower synthesizer:
    trust_min - workers' trust threshold,
    quiz_papers_n - number of papers for the quiz test,
    n_criteria - number of exclusion criteria,
    test_page - number of test questions per page,
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


def second_round():
    pass


def get_accuracy():
    pass


'''
gold_data = [[criteria_0 value, criteria_1 value,...],
             [criteria_0 value, criteria_1 value,...], [],..]

criteria_id value is a binary variable
each list in gold data presents criteria values for a row with id=list's index in the gold_data
'''
def generate_gold_data(n_criteria=3, test_page=1, papers_page=3, n_papers=30):
    gold_data = []
    pages_n = n_papers/papers_page
    tests_n = test_page * pages_n
    total_papers_n = tests_n + n_papers
    for row_id in range(total_papers_n):
        # decide if paper in/out of scope
        # bernoulli trial with p=0.5
        if random.randint(0, 1):
            paper_id_gold_data = [0]*n_criteria
        else:
            paper_id_gold_data = [random.randint(0, 1) for _ in range(n_criteria)]
        gold_data.append(paper_id_gold_data)
    return gold_data


# generate worker's accuracy
def get_worker_accuracy(trust_min, quiz_papers_n):
    n_init = int(trust_min*quiz_papers_n)
    possible_trust_values = [float(n)/quiz_papers_n for n in range(n_init, quiz_papers_n+1, 1)]
    worker_accuracy = (random.uniform(0.6, 1.) + random.choice(possible_trust_values))/2
    return worker_accuracy


'''
workers_judgment = [{row_id: [criteria_0 value, criteria_1 value,...],
                    row_id2: [criteria_0 value, criteria_1 value,...]}, {},..]

each element in 'workers_judgment' is judgments of a trustworthy worker who passed tests questions,
indexes of the workers_judgment' list present workers' id
'''
def first_round(trust_min, n_criteria, test_page, papers_page,
                n_pages, n_papers, budget, price_row, gold_data,
                judgment_min, judgment_max, cheaters_prop, quiz_papers_n):
    budget_rest = budget
    pages_n = n_papers / papers_page
    rows_page = test_page+papers_page
    workers_judgment = []
    for page_id in range(pages_n):
        for row_id_id in range(page_id*rows_page, page_id*rows_page+rows_page, 1):
            trust_judgment = 0
            while trust_judgment != judgment_min:
                worker_accuracy = get_worker_accuracy(trust_min, quiz_papers_n)
                pass
            # do_judgment()
            # is_passed_tests()
    pass



def synthesizer(trust_min=0.75, n_criteria=3, test_page=1, papers_page=3,
                n_pages=1, n_papers=30, budget=50, price_row=0.4,
                judgment_min=3, judgment_max=5, cheaters_prop=0.2, quiz_papers_n=4):
    gold_data = generate_gold_data(n_criteria=3, test_page=1, papers_page=3, n_papers=30)
    first_round(trust_min, n_criteria, test_page, papers_page,
                n_pages, n_papers, budget, price_row, gold_data,
                judgment_min, judgment_max, cheaters_prop, quiz_papers_n)




if __name__ == '__main__':
    synthesizer()
