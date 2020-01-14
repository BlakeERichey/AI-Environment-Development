from worker import Worker
import numpy as np

class Evolution:

  def __init__(self, generations, pop_size, elites, sharpness, goal, metric='reward'):
    self.generations = generations
    self.pop_size = pop_size
    self.elites = elites
    self.sharpness = sharpness
    self.goal = goal
    self.metric = metric

    self.workers = []

  def create_species(self, nn, mutations=1, patience=25, alpha=0.05):
    for i in range(self.pop_size):
      self.workers.append(Worker(nn, mutations, patience, alpha))

  def train(self, env, validate=False, render=False, return_worker=False):
    assert len(self.workers), "No Species Created."
    for gen in range(self.generations):
      ranked = []
      for i, worker in enumerate(self.workers):
        res, val = worker.fitness(env, self.sharpness, validate, render)
        ranked.append((i, res, val))

      ranked = sorted(ranked, key= lambda x: (x[1], x[2]), reverse=True)
      print("Gen:", gen, "Ranked:", ranked, '\n')
      if self.metric == 'reward':
        goal_met = ranked[0][1]>=self.goal
      else:
        goal_met = ranked[0][2]>=self.goal
      if gen != self.generations - 1 and not(goal_met):
        #Gen new weights
        mating_pool = self.selection(ranked)
        new_weights = []
        for i, worker in enumerate(mating_pool):
          if len(new_weights) < self.pop_size - self.elites:
            parent1 = self.workers[worker[0]]
            parent2 = self.workers[mating_pool[-i][0]]
            weights = parent1.breed(parent2)
            new_weights.append(weights)
        
        #determine if new mask is needed
        for i, tup in enumerate(ranked):
          worker_id = tup[0]
          self.workers[worker_id].add_rank(i, self.pop_size)

        #apply new weights and mutate
        for i, worker in enumerate(self.workers):
          if i > self.elites:
            worker.net.weights = new_weights[i-self.elites]
          worker.mutate()
      
      if goal_met:
        break
        
    best_worker = ranked[0][0] #id
    worker = self.workers[best_worker]
    if return_worker:
      return worker
    return worker.net

  
  def selection(self, ranked):
    mating_pool = []
    for i in range(self.elites):
      mating_pool.append(ranked[i])
    
    remaining = np.random.choice(len(ranked), len(ranked), replace=False)
    for i in remaining:
      worker = ranked[i]
      if worker not in mating_pool:
        mating_pool.append(worker)

    return mating_pool
    