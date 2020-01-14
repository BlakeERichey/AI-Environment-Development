from worker import Worker

class Evolution:

  def __init__(self, generations, pop_size, elites, sharpness, goal):
    self.generations = generations
    self.pop_size = pop_size
    self.elites = elites
    self.sharpness = sharpness
    self.goal = goal

    self.workers = []

  def create_species(nn, mutations, patience, alpha=0.05):
    for i in range(self.pop_size):
      self.workers.append(Worker(nn, mutation, patience, alpha))