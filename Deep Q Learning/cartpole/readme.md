Summary
---

OpenAI Gym's Cartpole problem is an environment where the goal is to train an 
agent to be able to balance a pole on a moving cart  

Here I present a Deep Q Learning Implementation to solving this problem. 

I use checkpoint saving through a custom function to save the best model at 
every epoch and an epislon greedy decision policy. I start with a high epsilon 
to give the agent the opportunity to explore the environment early, but quickly
reduce epislon to force the agent to strictly adhere to its best decision.

Results
---

Solution reached in 20 epochs/episodes

![Reward Plot](/Deep%20Q%20Learning/cartpole/results/rewards.png "Cumulative Reward per Epoch")  
*Cumulative rewards at each epoch during training*

Output of program:  
```
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 4)                 20        
_________________________________________________________________
dense_1 (Dense)              (None, 24)                120       
_________________________________________________________________
dense_output (Dense)         (None, 2)                 50        
=================================================================
Total params: 190
Trainable params: 190
Non-trainable params: 0
_________________________________________________________________
Starting training at 2019-08-17 17:09:34.507421
Action Decision Policy: eg
Epoch: 0/23 | Loss: 3.5970 | Steps 18 | Epsilon: 0.900 | Reward: 18.000 | Time: 2.0 seconds
Epoch: 1/23 | Loss: 1.8163 | Steps 46 | Epsilon: 0.450 | Reward: 46.000 | Time: 8.1 seconds
Epoch: 2/23 | Loss: 1.6197 | Steps 12 | Epsilon: 0.225 | Reward: 12.000 | Time: 9.7 seconds
Epoch: 3/23 | Loss: 1.2970 | Steps 19 | Epsilon: 0.113 | Reward: 19.000 | Time: 12.3 seconds
Epoch: 4/23 | Loss: 1.1842 | Steps 14 | Epsilon: 0.100 | Reward: 14.000 | Time: 14.2 seconds
Epoch: 5/23 | Loss: 0.6360 | Steps 26 | Epsilon: 0.100 | Reward: 26.000 | Time: 17.9 seconds
Epoch: 6/23 | Loss: 0.3956 | Steps 32 | Epsilon: 0.100 | Reward: 32.000 | Time: 22.3 seconds
Epoch: 7/23 | Loss: 0.0489 | Steps 200 | Epsilon: 0.100 | Reward: 200.000 | Time: 49.6 seconds
New best model reached: { 0.048894114792346954 200.0 }
Epoch: 8/23 | Loss: 1.5744 | Steps 200 | Epsilon: 0.100 | Reward: 200.000 | Time: 77.2 seconds
Epoch: 9/23 | Loss: 1.2132 | Steps 97 | Epsilon: 0.100 | Reward: 97.000 | Time: 90.7 seconds
Epoch: 10/23 | Loss: 2.2613 | Steps 30 | Epsilon: 0.100 | Reward: 30.000 | Time: 94.8 seconds
Epoch: 11/23 | Loss: 0.7509 | Steps 11 | Epsilon: 0.100 | Reward: 11.000 | Time: 96.4 seconds
Epoch: 12/23 | Loss: 0.0535 | Steps 21 | Epsilon: 0.100 | Reward: 21.000 | Time: 99.3 seconds
Epoch: 13/23 | Loss: 0.0161 | Steps 13 | Epsilon: 0.100 | Reward: 13.000 | Time: 101.2 seconds
Epoch: 14/23 | Loss: 1.6384 | Steps 12 | Epsilon: 0.100 | Reward: 12.000 | Time: 102.8 seconds
Epoch: 15/23 | Loss: 0.1589 | Steps 11 | Epsilon: 0.100 | Reward: 11.000 | Time: 104.3 seconds
Epoch: 16/23 | Loss: 0.0791 | Steps 200 | Epsilon: 0.100 | Reward: 200.000 | Time: 131.8 seconds
Epoch: 17/23 | Loss: 0.0829 | Steps 197 | Epsilon: 0.100 | Reward: 197.000 | Time: 158.9 seconds
Epoch: 18/23 | Loss: 0.0222 | Steps 200 | Epsilon: 0.100 | Reward: 200.000 | Time: 186.5 seconds
New best model reached: { 0.022194016724824905 200.0 }
Epoch: 19/23 | Loss: 0.0073 | Steps 200 | Epsilon: 0.100 | Reward: 200.000 | Time: 214.6 seconds
New best model reached: { 0.007333249785006046 200.0 }
Epoch: 20/23 | Loss: 1.6981 | Steps 11 | Epsilon: 0.100 | Reward: 11.000 | Time: 216.1 seconds
Epoch: 21/23 | Loss: 2.0274 | Steps 9 | Epsilon: 0.100 | Reward: 9.000 | Time: 217.4 seconds
Epoch: 22/23 | Loss: 0.0324 | Steps 162 | Epsilon: 0.100 | Reward: 162.000 | Time: 240.3 seconds
Epoch: 23/23 | Loss: 0.0034 | Steps 15 | Epsilon: 0.100 | Reward: 15.000 | Time: 242.4 seconds
Weights saved to: ./results/cartpole.h5
￼
Added Dense layer with 24 nodes.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 4)                 20        
_________________________________________________________________
dense_3 (Dense)              (None, 24)                120       
_________________________________________________________________
dense_output (Dense)         (None, 2)                 50        
=================================================================
Total params: 190
Trainable params: 190
Non-trainable params: 0
_________________________________________________________________
Successfully loaded weights from: ./results/best_model.h5
Evaluating... Starting at: 2019-08-17 17:13:37.589919
Average Reward: 200.0
Evaluating... Starting at: 2019-08-17 17:14:10.770983
Epoch: 0/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 5.6 seconds
Epoch: 1/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 11.2 seconds
Epoch: 2/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 16.9 seconds
Epoch: 3/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 22.3 seconds
Epoch: 4/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 27.7 seconds
Epoch: 5/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 33.5 seconds
Epoch: 6/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 38.7 seconds
Epoch: 7/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 43.3 seconds
Epoch: 8/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 48.0 seconds
Epoch: 9/9 | Steps 200 | Cumulative Reward: 200.0 | Time: 53.5 seconds
```