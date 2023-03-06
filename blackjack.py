#!/usr/bin/env python
# coding: utf-8

# In[4]:


from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import random

"""
WRITE THIS FUNCTION
"""
def value_iteration(
    V0: npt.NDArray, 
    lr: float, 
    gamma:float, 
    epsilon: float=1e-12
    ) -> npt.NDArray:
    while True:
      update_V = np.zeros(V0.size)
      for i in range(0, V0.size):
        prob = transition_draw(i)
        card = 1
        draw = 0
        for p in prob:
          if i+card <= 21:
            draw += p * (lr + gamma * V0[i+card])
            card += 1
          else:
            draw += p * (lr + gamma * 0)
        stop = i
        update_V[i] = max(draw, stop)
      diff = max(np.absolute(update_V - V0))
      if diff < epsilon:
        return V0
      V0 = update_V


def transition_draw(s):
  if s <= 12:
    return [1/13]*9 + [4/13]
  else:
    return [1/13]*(21 - s) + [(s - 8)/13]





# In[12]:


"""
WRITE THIS FUNCTION
"""
def value_to_policy(V: npt.NDArray, lr: float, gamma: float) -> npt.NDArray:
  policy = np.zeros(V.size)
  for i in range(0, V.size):
    prob = transition_draw(i)
    draw = 0
    card = 1
    for p in prob:
      if i + card <= 21:
        draw += p * (lr + gamma * V[i + card])
        card += 1
      else:
        draw += p * (lr + gamma * 0)
    stop = i
    if draw > stop:
      policy[i] = 1
  
  return policy


# In[13]:


def draw() -> int:
  probs = 1/13*np.ones(10)
  probs[-1] *= 4
  return np.random.choice(np.arange(1,11), p=probs)

def epsilon_greedy (epsilon, Q, state):
  num = random.random()
  if Q[state][0] < Q[state][1]:
    min = 0
    max = 1
  else:
    min = 1
    max = 0
  if num < epsilon: #explore
    return np.random.randint(2)
  else: #exploit
    return max

"""
WRITE THIS FUNCTION
"""
def Qlearn(
    Q0: npt.NDArray, 
    lr: float, 
    gamma: float, 
    alpha: float, 
    epsilon: float, 
    N: int
    ) -> Tuple[npt.NDArray, npt.NDArray]:
  record = np.zeros((N,3))
  state = 0
  for i in range(0, N):
    action = epsilon_greedy(epsilon, Q0, state)
    if action == 0:
      r = state
      target = r
      successor = 0
    elif action == 1:
      r = lr
      successor = state + draw()
      if successor <= 21:
        target = r + gamma * max(Q0[successor][0], Q0[successor][1])
      else:
        target = r
    record[i][0] = state
    record[i][1] = action
    record[i][2] = r
    Q0[state, action] = Q0[state, action] + alpha * (target - Q0[state][action])
    if successor > 21 or action == 0:
      state = 0
    else:
      state = successor

  
  return Q0, record


# In[14]:


def RL_analysis():
  lr, gamma, alpha, epsilon, N = 0, 1, 0.1, 0.1, 10000
  visits = np.zeros((22,6))
  rewards = np.zeros((N,6))
  values = np.zeros((22,6))

  for i in range(6):
    _, record = Qlearn(np.zeros((22,2)), lr, gamma, alpha, epsilon, 10000*i)
    vals, counts = np.unique(record[:,0], return_counts=True)
    visits[vals.astype(int),i] = counts
    _, record = Qlearn(np.zeros((22,2)), lr, gamma, alpha, 0.2*i, N)
    rewards[:,i] = record[:,2]
    vals, _ = Qlearn(np.zeros((22,2)), lr, gamma, min(0.2*i+0.1,1), epsilon, N)
    values[:,i] = np.max(vals, axis=1)

  plt.figure()
  plt.plot(visits)
  plt.legend(['N=0', 'N=10k', 'N=20k', 'N=30k' ,'N=40k', 'N=50k'])
  plt.title('Number of visits to each state')
  plt.show()

  plt.figure()
  plt.plot(np.cumsum(rewards, axis=0))
  plt.legend(['e=0.0', 'e=0.2', 'e=0.4' ,'e=0.6', 'e=0.8', 'e=1.0'])
  plt.title('Cumulative rewards received')
  plt.show()

  plt.figure()
  plt.plot(values)
  plt.legend(['a=0.1' ,'a=0.3', 'a=0.5', 'a=0.7', 'a=0.9', 'a=1.0'])
  plt.title('Estimated state values');
  plt.show()




if __name__ == "__main__":

  v = np.zeros(22)
  rl = 1
  gamma = 1
  V_optimal = value_iteration(v, rl, gamma)
  print(V_optimal)
  policy = value_to_policy(V_optimal, rl, gamma)
  print(policy)
  plt.plot(V_optimal)
  plt.show()
  plt.plot(policy)
  plt.show()

  RL_analysis()
