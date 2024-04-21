import heapq
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans


class Solution:
    def __init__(self, x, bd, fitness, centroid=None):
        self.x = x
        self.bd = bd
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None

    def __lt__(self, other):
        return self.fitness < other.fitness


class Archive:
    def __init__(self, filename):
        self.filename = filename
        self.best_solution = None
        self.random_state = 0

    @property
    def coverage(self):
        raise NotImplementedError

    def info(self) -> dict:
        raise NotImplementedError

    @property
    def max_fitness(self):
        if self.best_solution is None:
            return float('-inf')
        else:
            return self.best_solution.fitness

    def add_to_archive(self, solution: Solution):
        raise NotImplementedError

    def top_solutions(self, k):
        raise NotImplementedError

    def random_solutions(self, k):
        raise NotImplementedError

    def save(self):
        torch.save(self, self.filename)


class FitnessPriorityArchive(Archive):
    def __init__(self, capacity, filename='archive.pkl'):
        super().__init__(filename)
        self.capacity = capacity
        self.queue = []
        self.best_solution = None

    def info(self) -> dict:
        qd_score = 0
        min_fitness = float('inf')
        for solution in self.queue:
            qd_score += solution.fitness
            min_fitness = min(min_fitness, solution.fitness)
        return dict(
            coverage=self.coverage,
            max_fitness=self.max_fitness,
            min_fitness=min_fitness,
            qd_score=qd_score
        )

    @property
    def coverage(self):
        return len(self.queue)

    def add_to_archive(self, solution: Solution):
        if self.coverage == self.capacity:
            worst_solution = self.queue[0]
            if solution.fitness <= worst_solution.fitness:
                return False
            heapq.heappop(self.queue)
        heapq.heappush(self.queue, solution)
        if solution.fitness > self.max_fitness:
            self.best_solution = solution
        return True

    def top_solutions(self, k):
        return heapq.nlargest(k, self.queue)

    def random_solutions(self, k):
        return np.random.choice(self.queue, k, replace=False)


class MAPElitesArchive(Archive):
    def __init__(self, shape=(10, 10), filename='archive.pkl'):
        super().__init__(filename)
        self.shape = np.array(shape, dtype=int)
        self.capacity = np.prod(shape)
        self.table = np.zeros(shape=shape, dtype=object)
        self.table[:] = None
        self.indices = []
        self.best_solution = None

    def info(self) -> dict:
        qd_score = 0
        min_fitness = float('inf')
        for index in self.indices:
            solution = self.table[index]
            qd_score += solution.fitness
            min_fitness = min(min_fitness, solution.fitness)
        return dict(
            coverage=self.coverage,
            max_fitness=self.max_fitness,
            min_fitness=min_fitness,
            qd_score=qd_score
        )

    @property
    def qd_score(self, min_fitness=None):
        if min_fitness is None:
            return self._qd_score
        else:
            return self._qd_score + self.coverage * min_fitness

    @property
    def max_fitness(self):
        if self.best_solution is None:
            return float('-inf')
        else:
            return self.best_solution.fitness

    @property
    def coverage(self):
        return len(self.indices)

    def get_cell_index(self, bd):
        return tuple(np.minimum((self.shape * bd).astype(int), self.shape - 1))

    def add_to_archive(self, solution: Solution):
        # get cell index
        index = self.get_cell_index(solution.bd)
        elite: Optional[Solution] = self.table[index]
        if elite is None or elite.fitness < solution.fitness:
            if elite is None:
                self.indices.append(index)
            if solution.fitness > self.max_fitness:
                self.best_solution = solution
            self.table[index] = solution
            return True
        return False

    def random_solutions(self, k):
        random_results = np.random.randint(len(self.indices), size=k)
        results = [self.table[self.indices[i]] for i in random_results]
        return results

    def top_solutions(self, k):
        # solutions = [self.table[index] for index in self.indices]
        # solutions.sort(reverse=True)
        # return solutions[:k]
        return self.cluster_based_selection(k)

    def cluster_based_selection(self, k):
        bds = np.array([self.table[index].bd for index in self.indices])
        cs = KMeans(n_init=1, n_clusters=k, random_state=self.random_state).fit_predict(bds)
        self.random_state += 1
        results = [None] * k
        for c, index in zip(cs, self.indices):
            if results[c] is None or results[c].fitness < self.table[index].fitness:
                results[c] = self.table[index]
        return results

    def save(self):
        torch.save(self, self.filename)
