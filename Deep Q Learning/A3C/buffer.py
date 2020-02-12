import random
class ReplayBuffer:

  def __init__(self,size,):
    self.size = size
    self.length = 0
    self.memory = []
  
  def remember(self, cur_state, action, reward, new_state, done):
    '''
      Remembers a new sample while removing older samples exceeding 
      replay buffer size
    '''
    
    experience = [cur_state, action, reward, new_state, done]
    if self.length == self.size: #saves computing len for longer arrays
      self.memory = self.memory[1:]
      self.memory.append(experience)
    else:
      self.memory.append(experience)
      self.length+=1
  
  def get_batch(self, batch_size=1):
    return random.sample(self.memory, batch_size)

