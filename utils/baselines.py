import numpy as np

def naive_baseline(state): # Example Baseline from lecture 4 (for inspiration)
  angle = state[2]
  value = 100*(0.25-angle**2) # TO BE CHANGED USING YOUR BASELINE
  return value

def baseline_1(state): # TO BE CHANGED USING YOUR BASELINE 1
  cart_position = state[0]   # TODO
  cart_velocity = state[1]
  angle = state[2]
  angular_velocity = state[3]

  value = - 10*np.abs(angle/0.2) - 0.5*np.abs(angular_velocity)

  return value # TODO

def baseline_2(state): # TO BE CHANGED USING YOUR BASELINE 2
  cart_position = state[0]   # TODO
  cart_velocity = state[1]
  angle = state[2]
  angular_velocity = state[3]

  value = - 10*np.abs(angle/0.2) - 0.5*np.abs(cart_position/2)
  return value # TODO


