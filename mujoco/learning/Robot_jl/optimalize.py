import random

import numpy as np


pop_number = 100
pop = []
def g_gene(iter,pop_number):

    for i in range(pop_number):
        a = np.random.rand(18)
        pop.append(a)
    return pop

def hc(pop):

    for i in range(len(pop)):
        # idx1 = random.randint(0,2)
        # idx2 = random.randint(0,5)
        # pop[i][idx1][idx2] = random.random()
        idx = random.randint(0,17)
        pop[i][idx] = random.random()

    return pop
