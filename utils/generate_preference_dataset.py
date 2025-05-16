import numpy as np

def generate_trajectories(policy, n_trajectories, env, max_t=int(1000), seed=0, dim_state=4, print_every=10):
    """
    Runs multiple trajectories with a specified policy and seed. Adapt dim_state according to the environment
    Arguments:
     - policy, 
     - n_trajectories, 
     - env, 
     - max_t=int(1000), 
     - seed=0, 
     - dim_state=4,
     - print_every=10

    Returns:
     - trajectories_rewards, [np.array(n_trajectories)]
     - trajectories_states [np.array((n_trajectories, max_t+int(1), dim_state))]
     - trajectories_actions [np.array((n_trajectories, max_t+int(1)))]
    """
    trajectories_states = np.zeros((n_trajectories, max_t+int(1), dim_state))
    trajectories_actions = np.zeros((n_trajectories, max_t+int(1)))
    trajectories_rewards = np.zeros(n_trajectories)
    for traj_index in range(n_trajectories):
        saved_log_probs = []
        rewards = []

        # start trajectory
        state,_ = env.reset(seed=seed)
        trajectories_states[traj_index,0] = state
        # Collect trajectory
        for t in range(1,max_t+1):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            trajectories_actions[traj_index, t-1] = action
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, info = env.step(action)
            #save state
            trajectories_states[traj_index, t,:] = state[:]
            rewards.append(reward)
            if done:
                break
        action, log_prob = policy.act(state)
        trajectories_actions[traj_index, max_t] = action
        # Calculate trajectory reward
        trajectories_rewards[traj_index] = sum(rewards)
        if traj_index % print_every == 0:
            print('Trajectory: ', traj_index)
    return trajectories_rewards, trajectories_states, trajectories_actions