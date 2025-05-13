import numpy as np

def generate_trajectories(policy, n_trajectories, env, max_t=int(1000), seed=0, dim_state=4):
    trajectories_states = np.zeros((n_trajectories, max_t+int(1), dim_state))
    trajectories_rewards = np.zeros(n_trajectories)
    for traj_index in range(n_trajectories):
        saved_log_probs = []
        rewards = []
        state,_ = env.reset(seed=seed)
        trajectories_states[traj_index,0] = state
        # Collect trajectory
        for t in range(1,max_t+1):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, info = env.step(action)
            #save state
            trajectories_states[traj_index, t,:] = state[:]
            rewards.append(reward)
            if done:
                break
        # Calculate trajectory reward
        trajectories_rewards[traj_index] = sum(rewards)
    return trajectories_rewards, trajectories_states