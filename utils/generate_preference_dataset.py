import numpy as np
from classes import Policy
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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




def dataset(n_trajectories, max_t, seed, pi1_path, pi2_path, env, dim_state):
    preferences = np.zeros(n_trajectories)


    pi1 = Policy().to(device)
    pi1.load_state_dict(torch.load(pi1_path, weights_only=True))

    pi2 = Policy().to(device)
    pi2.load_state_dict(torch.load(pi2_path, weights_only=True))


    trajectories_rewards_pi1, trajectories_states_pi1, trajectories_actions_pi1 = generate_trajectories(pi1, n_trajectories, env=env, max_t=max_t, seed = seed, dim_state=dim_state)

    trajectories_rewards_pi2, trajectories_states_pi2, trajectories_actions_pi2 = generate_trajectories(pi2, n_trajectories, env=env, max_t=max_t, seed = seed, dim_state=dim_state)

    preferences = np.exp(trajectories_rewards_pi1 - 500) / (np.exp(trajectories_rewards_pi1-500) + np.exp(trajectories_rewards_pi2-500))

    dataset = [trajectories_states_pi1, trajectories_actions_pi1, trajectories_states_pi2, trajectories_actions_pi2, preferences]
    return dataset


