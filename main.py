import numpy as np
from random import random, shuffle, randint
from itertools import tee

alpha = -10
beta = 10
n = 3

def f(x):
    global n
    # return sum((x[i])**2 for i in range(n))
    return sum((x[i])**2 - 10*np.cos(2*np.pi*x[i]) +10 for i in range(n))

def fitness(x):
    return 1/f(x)


def new_coord(coord, r):
    u = random()
    if u > 0.5:
        coord = coord + np.random.uniform(0, beta - coord) * r
    else:
        coord = coord - np.random.uniform(0, coord - alpha) * r
    return coord

def generate_population(Np):
    global alpha, beta, n
    pop = []
    for i in range(Np):
        p = np.random.uniform(alpha, beta, 3)
        # p = np.array([(beta - alpha)*p0[j] + alpha for j in range(3)])
        pop.append(p)
        # print(p)
    return pop

def immune(f, K=100, Np=100, s=10, d=40, mu=0.7, r=0.3):
    global n
    pop = generate_population(Np)
    # print(pop)
    for _ in range(K):
        pop.sort(key=fitness)
        clone_sources = pop[s:]
        for i, p in enumerate(clone_sources):
            clones = [p for _ in range(max(1, int(mu * Np/(i+1))))]
            # print(clones[0])
            for clone in clones:
                for j in range(n):
                    mutated = new_coord(clone[j], r)
                    while not(alpha <= mutated <= beta):
                        mutated = new_coord(clone[j], r)
                    clone[j] = mutated
            # print(clones[0])
            best = min(clones, key=f)
            pop[s+i] = min(pop[s+i], best, key=f)
            # print(pop[s+i])
        pop[:d] = generate_population(d)
    return min(pop, key=f)


def main():
    p = immune(f)
    print(p, f(p))

if __name__ == '__main__':
    main()
