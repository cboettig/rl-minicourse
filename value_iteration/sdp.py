import numpy as np
from scipy.stats import poisson
from plotnine import ggplot, aes, geom_line, geom_point
import pandas as pd

# state space limit
Nmax = 250
# Vector of all possible states
states = range(Nmax)

# Vector of actions: rate of the population that can be removed, ranging #from 0 to 1
actions = np.linspace(0, 1, Nmax+1)

# Population growth rate
r = 0.25
K = 100
# Function for the exponential growth of the dynamic model
def dynamic(x, action):
    y = x * (1 - action)
    x_t1 = x + y * r * (1 - y / K)

    if(x_t1 < 0): 
      x_t1 = 0
    return(x_t1)

# Utility function
def get_utility(x, a):
  return(a * x)

# Fill in the transition and utility matrix
transition = np.zeros([len(actions), len(states), len(states)])
utility = np.zeros([len(states), len(actions)])

for k in states:
  for i in range(len(actions)):
    # Calculate the transition state at the next step, given the #current state k and the harvest Hi
    nextpop = dynamic(k, actions[i])
    transition[i, k, :] = poisson.pmf(states, nextpop)
    # We need to correct this density for the final capping state
    transition[i, Nmax-1, k] = 1.0 - np.sum(transition[i, k, range(Nmax-2)])
    transition[i, k, :] = transition[i, k, :] / np.sum(transition[i, k, :])
    utility[k,i] = get_utility(nextpop, actions[i])

# Discount factor
discount = 0.9


# let's write our own value iteration
Vt = np.zeros(len(states)) # Value-to-go 
# Optimal policy vector
D = np.zeros(len(states))

# Time horizon
Tmax = 300

# The backward iteration consists in storing action values in the vector Vt which is the maximum of
# utility plus the future action values for all possible next states. Knowing the final action
# values, we can then backwardly reset the next action value Vtplus to the new value Vt. We start
# The backward iteration at time T-1 since we already defined the action #value at Tmax.
for t in range(Tmax):
  # We define a matrix Q that stores the updated action values for #all states (rows)
  # actions (columns)
  Q = np.zeros([len(states), len(actions)])
  for i in range(len(actions)):
    # For each harvest rate we fill for all states values (row)
    # the ith column (Action) of matrix Q
    Q[:,i] = utility[:,i] + discount * np.dot(transition[i,:,:], Vt)
  # Find the optimal action value at time t is the maximum of Q
  Vt = np.amax(Q, 1)

# Find optimal action for each state
for k in states:
  # We look for each state which column of Q corresponds to the
  # maximum of the last updated value
  # of Vt (the one at time t+1).
  # (if there is more than one optimal value we chose the minimum action)
  D[k] = actions[np.min(np.where(Q[k,:] == Vt[k])[0])]

# Plot the result
df = pd.DataFrame()
df['states'] = states[0:100]
df['policy'] = D[0:100]
df['escapement'] = df.states - df.policy * df.states
ggplot(df) + geom_line(aes("states", "policy"))
ggplot(df) + geom_line(aes("states", "escapement"))



### Verify that standard implementations agree with us:

# pip install pymdptoolbox
from mdptoolbox import mdp
vi = mdp.ValueIteration(transition, utility, discount)
vi.run()

df = pd.DataFrame()
df['states'] = states[1:100]
df['policy'] = vi.policy[1:100]
ggplot(df) + geom_line(aes("states", "policy"))
