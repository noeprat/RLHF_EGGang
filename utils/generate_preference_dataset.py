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

def generate_preference_dataset(pi1, pi2, dataset_size, env, max_t=int(999), seed=0, dim_state=4, print_every=10, sampling_method='pi1_pi2_trajectories'):
    """
    Generate a preference dataset of size n_trajectories*(max_t+1), using pi1 and pi2. Careful, dataset_size must be a multiple of (max_t+1)
    Arguments:
    pi1, 
    pi2, 
    dataset_size, 
    env, 
    max_t=int(1000), 
    seed=0, 
    dim_state=4, 
    print_every=10,
    sampling_method, str for how the states are sampled

    Returns:
    states, np.array(dataset_size, dim_state)
    preferred_actions, np.array(dataset_size)
    rejected_actions, np.array(dataset_size)
    """
    n_trajectories = dataset_size // (max_t+1)
    states = np.zeros((dataset_size, dim_state))
    preferred_actions = np.zeros(dataset_size)
    rejected_actions = np.zeros(dataset_size)

    if sampling_method == 'pi1_pi2_trajectories':

        print("\n First half \n")
        _, trajectories_states, trajectories_actions = generate_trajectories(pi1, n_trajectories//2, env, max_t, seed, dim_state, print_every)
        for traj_index in range(n_trajectories//2):
            for t in range(max_t+1):
                states[traj_index*(int(max_t)+1)+t,:] = trajectories_states[traj_index, t, :]
                preferred_actions[traj_index*(max_t+1)+t] = trajectories_actions[traj_index, t]
                rejected_actions[traj_index*(max_t+1)+t] = pi2.act(trajectories_states[traj_index,t])[0]

        print("\n Second half \n")
        _, trajectories_states, trajectories_actions = generate_trajectories(pi2, n_trajectories//2, env, max_t, seed, dim_state, print_every)
        for traj_index in range(n_trajectories//2):
            for t in range(max_t+1):
                states[(traj_index+n_trajectories//2)*(max_t+1)+t,:] = trajectories_states[traj_index, t,:]
                preferred_actions[traj_index*(max_t+1)+t] = pi1.act(trajectories_states[traj_index,t])[0]
                rejected_actions[traj_index*(max_t+1)+t] = trajectories_actions[traj_index, t]
    elif sampling_method == 'uniform_cartpole':
        states[:,0] = np.random.uniform(low= -4.8, high= 4.8, size=dataset_size)
        states[:,1] = np.random.standard_normal(dataset_size)
        states[:,2] = np.random.uniform(low= -0.418, high= 0.418, size=dataset_size)
        states[:,3] = np.random.standard_normal(dataset_size)
    
        preferred_actions = np.array([pi1.act(state)[0] for state in states])
        rejected_actions = np.array([pi2.act(state)[0] for state in states])


    return states, preferred_actions, rejected_actions