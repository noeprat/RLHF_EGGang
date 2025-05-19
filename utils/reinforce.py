import numpy as np
from collections import deque

import torch
import os
PATH = os.path.abspath(os.getcwd())

def reinforce_rwd2go(policy, optimizer, seed, env, early_stop=False, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    """
    Implements Reward-to-go and returns only the scores array (not used so far)
    Arguments:
     - policy, optimizer, seed, env, early_stop=False, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100
    Returns:
     - scores
    """
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset(seed=seed)
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum([discounts[j]*rewards[j+t] for j in range(len(rewards)-t) ]) for t in range(len(rewards))]

        # Calculate the loss
        policy_loss = []
        for i in range(len(saved_log_probs)):
            log_prob = saved_log_probs[i]
            G = rewards_to_go[i]
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * G)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if early_stop and np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            break
    return scores


def reinforce_rwd2go_baseline(policy, optimizer, seed, baseline, env, target_score=100, early_stop=False, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, save_models_every=False):
    """
    Implements REINFORCE with a specified baseline and returns the scores array, the terminal policy and a checkpoint policy (earning +- 10% of the target score)
    Arguments:
     - policy, the initial policy [class(Policy)]
     - optimizer, the chosen optimizer
     - seed, [int]
     - baseline, the baseline function [function]
     - env, 
     - target_score = 100, [float]
     - early_stop=False, 
     - n_episodes=1000, 
     - max_t=1000, 
     - gamma=1.0, 
     - print_every=100
    Returns:
     - scores
     - max_policy
     - checkpoint_policy
    """
    max_policy_reward = 0
    scores_deque = deque(maxlen=100)
    scores = []
    for e in range(n_episodes+1):
        saved_log_probs = []
        rewards = []
        baseline_values = []
        state,_ = env.reset(seed=seed)
        # Collect trajectory
        for t in range(max_t):
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            baseline_values.append(baseline(state))
            if done:
                break
        # Calculate total expected reward
        policy_reward = sum(rewards)
        scores_deque.append(policy_reward)
        scores.append(policy_reward)

        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        rewards_to_go = [sum([discounts[j]*rewards[j+t] for j in range(len(rewards)-t) ]) for t in range(len(rewards))]

        # Calculate the loss
        policy_loss = []
        for i in range(len(saved_log_probs)):
            log_prob = saved_log_probs[i]
            G_centered = rewards_to_go[i] - baseline_values[i]
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * G_centered)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
            print('Episode {}\tCurrent Score: {:.2f}'.format(e, policy_reward))
            if save_models_every:
                model_path = os.path.join(PATH, 'saved_policies', 'model_{}.pt'.format(e))
                torch.save(policy.state_dict(), model_path, _use_new_zipfile_serialization=False)


        if early_stop and np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            break

        if policy_reward >= 0.90*target_score and policy_reward <= 1.10 * target_score:
            model_path = os.path.join(PATH, 'saved_policies', 'model_{}.pt'.format(e))
            torch.save(policy.state_dict(), model_path, _use_new_zipfile_serialization=False)
            print('Episode {}\tCurrent Score: {:.2f}'.format(e, policy_reward))

        if policy_reward >= max_policy_reward:
            max_policy_reward = policy_reward
            model_path = os.path.join(PATH, 'saved_policies', 'model_{}.pt'.format(e))
            torch.save(policy.state_dict(), model_path, _use_new_zipfile_serialization=False)
            print('Episode {}\tCurrent Score: {:.2f}'.format(e, policy_reward))

    return scores #, max_policy, checkpoint_policy