Goal
===

To create a dynamic reusable class for NN development and testing

Directory Structure
---

* CNN  
A project focused on implementing a dynamic cnn creation class. Used to solve 
mnist data on numbers using unsupervised learning.  

* Cartpole  
A project to test the ability of the DQAgent class to test if it can solve the 
[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) problem from OpenAI Gym.  

* MountainCar  
A significantly harder problem implementing DQAgent attempting to solve the 
[MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) problem

* DQAgent  
Dynamic class for NN generation

Test
---

Here are some known environments the AI class been used to successfully solve.

* Cartpole  
Solution reached in 300 episodes

* MountainCar  
Solution reached in 11000 episodes

Requirements
---
gym, numpy, keras, tensorflow  

Todo
---

Add Eligibility Trace to better enable an AI to have long-term goals  
Add target/training network implementation
Add rnn for cnns