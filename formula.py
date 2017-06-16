import itertools
import pandas as pd


def get_loss(order, criteria_power, criteria_acc, CR):
    pfi = criteria_power[-1] * (1 - criteria_acc[-1])
    for j in order[:-1]:
        pfi_j = criteria_power[j] * (1 - criteria_acc[j])
        pti_j = (1 - criteria_power[j]) * criteria_acc[j]
        pin_j = pfi_j + pti_j
        pfi *= pin_j

    pfe_0 = (1 - criteria_power[0]) * (1 - criteria_acc[0])
    pfe = pfe_0
    for i in order[1:]:
        m = 1.
        pfe_i = (1 - criteria_power[i]) * (1 - criteria_acc[i])
        for j in order[:i+1]:
            pfi_j = criteria_power[j] * (1 - criteria_acc[j])
            pti_j = (1 - criteria_power[j]) * criteria_acc[j]
            pin_j = pfi_j + pti_j
            m *= pin_j
        pfe += pfe_i * m

    loss = CR * pfe + pfi
    return loss


def get_cost(order, criteria_power, criteria_acc):
    cost = 1.
    for i in order[:-1]:
        m = 1.
        for j in order[:i+1]:
            pfi_j = criteria_power[j] * (1 - criteria_acc[j])
            pti_j = (1 - criteria_power[j]) * criteria_acc[j]
            pin_j = pfi_j + pti_j
            m *= pin_j
        cost += m
    return cost


if __name__ == '__main__':
    CR = 1
    # criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_power = [0.14, 0.14, 0.14, 0.9]
    # criteria_acc = [0.6, 0.7, 0.8, 0.9]
    criteria_acc = [0.7, 0.7, 0.7, 0.9]
    criteria_num = len(criteria_power)

    print '----------------------------------'
    for cr_id in range(criteria_num):
        print 'cr_id: {} | power: {} | cr_acc: {}'.format(cr_id, criteria_power[cr_id], criteria_acc[cr_id])
    print '----------------------------------'

    orders = itertools.permutations(range(criteria_num))
    data = []
    for order in orders:
        loss = get_loss(order, criteria_power, criteria_acc, CR)
        cost = get_cost(order, criteria_power, criteria_acc)
        print '{} | loss: {} | cost: {}'.format(order, loss, cost)
        data.append([order, loss, cost])
    pd.DataFrame(data, columns=['order', 'loss', 'cost']). \
        to_csv('output/data/loss_cost_test.csv', index=False)
