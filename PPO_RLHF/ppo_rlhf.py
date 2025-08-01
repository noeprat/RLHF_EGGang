"""
	The file contains the PPO class to train with.
"""

import gymnasium as gym
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions import MultivariateNormal, Categorical
import os

from PPO_RLHF.networks import ActorNetwork, CriticNetwork, RewardModel

class PPORLHF:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, env, preference_data, seed=None, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				env - the environment to train on.
				preference_data_path - path to the preference data file (optional)
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) in [gym.spaces.Box, gym.spaces.Discrete])
		assert(type(env.action_space) in [gym.spaces.Box, gym.spaces.Discrete])

        # Initialize seeds
		if seed is not None:
			torch.manual_seed(seed)
			np.random.seed(seed)
			self.seed = seed

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		
		# Handle both discrete and continuous action spaces
		if isinstance(env.action_space, gym.spaces.Discrete):
			self.action_dim = env.action_space.n
		else:
			self.action_dim = env.action_space.shape[0]  # For continuous actions
		
		# Set observation dimension from observation space
		self.obs_dim = env.observation_space.shape[0]

		# Initialize the actor, critic, and reward model
		self.actor = ActorNetwork(self.action_dim, self.obs_dim, alpha=self.lr)
		self.critic = CriticNetwork(self.obs_dim, alpha=self.lr)
		self.reward_model = RewardModel(self.obs_dim, self.action_dim, alpha=self.lr)
		
		# Get the device from the actor network
		self.device = self.actor.device
		
		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(self.device)
		self.cov_mat = torch.diag(self.cov_var)
		
		# Initialize optimizers
		self.actor_optim = self.actor.optimizer
		self.critic_optim = self.critic.optimizer

		# Load preference data if provided
		self.preference_data = preference_data
		
		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'reward_model_losses': [], # losses of reward model
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {total_timesteps} timesteps ...")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		
		# First train the reward model if we have preference data
		if self.preference_data is not None:
			self.train_reward_model()
			self.reward_model.eval()
		
		while t_so_far < total_timesteps:
			# Collecting our batch simulations
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()

			# Normalizing advantages
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

		# Save models at the end of training
		print("\nTraining completed. Saving final models...")
		models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
		os.makedirs(models_dir, exist_ok=True)
		torch.save(self.actor.state_dict(), os.path.join(models_dir, 'ppo-rlhf_actor.pth'))
		torch.save(self.critic.state_dict(), os.path.join(models_dir, 'ppo-rlhf_critic.pth'))

	def rollout(self):
		"""
			This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. sNote that obs is short for observation. 
			obs, _ = self.env.reset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)
				obs, _, terminated, truncated, _ = self.env.step(action)

				# Replace env reward with reward model prediction
				obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
				if isinstance(self.env.action_space, gym.spaces.Discrete):
					# One-hot encode discrete action
					action_onehot = np.eye(self.action_dim)[int(action)]
					action_tensor = torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0).to(self.device)
				else:
					# Use continuous action as is
					action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
				with torch.no_grad():
					r_hat = self.reward_model(obs_tensor, action_tensor).item()
				# Scale reward using tanh to prevent extreme values while preserving sign
				r_hat = torch.tanh(torch.tensor(r_hat)).item()
				# Don't really care about the difference between terminated or truncated in this, so just combine them
				done = terminated | truncated

				# Track recent reward, action, and action log probability
				ep_rews.append(r_hat)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					break
			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
		batch_rtgs = self.compute_rtgs(batch_rews).to(self.device)

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Convert observation to tensor and move to device
		obs = torch.tensor(obs, dtype=torch.float).to(self.device)
		
		# Query the actor network for action probabilities
		self.actor.eval()
		action_probs = self.actor(obs)
		self.actor.train()
		
		if isinstance(self.env.action_space, gym.spaces.Discrete):
			# For discrete action spaces
			dist = Categorical(action_probs)
			action = dist.sample()
			log_prob = dist.log_prob(action)
			return action.item(), log_prob.detach()
		else:
			# For continuous action spaces
			dist = MultivariateNormal(action_probs, self.cov_mat)
			action = dist.sample()
			log_prob = dist.log_prob(action)
			return action.detach().cpu().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		batch_obs = batch_obs.to(self.device)
  
    # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		self.critic.eval()
		V = self.critic(batch_obs).squeeze()
		self.critic.train()

		# Calculate the log probabilities of batch actions using most recent actor network.
		self.actor.eval()
		action_probs = self.actor(batch_obs)
		self.actor.train()
		
		if isinstance(self.env.action_space, gym.spaces.Discrete):
			# For discrete action spaces
			dist = Categorical(action_probs)
			log_probs = dist.log_prob(batch_acts)
		else:
			# For continuous action spaces
			dist = MultivariateNormal(action_probs, self.cov_mat)
			log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# RLHF-specific hyperparameters
		self.reward_model_lr = 0.0003                   # Learning rate for reward model
		self.reward_model_epochs = 30                   # Number of epochs to train reward model

		# Change any default values to custom values for specified hyperparameters
		for param, value in hyperparameters.items():
			setattr(self, param, value)

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.cpu().float().mean().item() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		#print(flush=True)
		#print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		#print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		#print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		#print(f"Average Loss: {avg_actor_loss}", flush=True)
		#print(f"Timesteps So Far: {t_so_far}", flush=True)
		#print(f"Iteration took: {delta_t} secs", flush=True)
		#print(f"------------------------------------------------------", flush=True)
		#print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger['reward_model_losses'] = []

	def train_reward_model(self):
		"""
		Train the reward model using the preference dataset
		"""
		self.reward_model.train()
		print(f"Training reward model for {self.reward_model_epochs} epochs...")
		
		# Split data into train and validation sets (80-20 split)
		n_samples = len(self.preference_data)
		indices = np.random.permutation(n_samples)
		train_size = int(0.8 * n_samples)
		train_indices = indices[:train_size]
		val_indices = indices[train_size:]
		
		# Create batches
		batch_size = 32
		n_batches = (train_size + batch_size - 1) // batch_size
		
		best_val_loss = float('inf')
		patience = 5
		patience_counter = 0
		
		for epoch in tqdm(range(self.reward_model_epochs)):
			total_train_loss = 0
			total_val_loss = 0
			
			# Training
			self.reward_model.train()
			for batch_idx in range(n_batches):
				start_idx = batch_idx * batch_size
				end_idx = min((batch_idx + 1) * batch_size, train_size)
				batch_indices = train_indices[start_idx:end_idx]
				
				self.reward_model.optimizer.zero_grad()
				batch_loss = 0
				
				for idx in batch_indices:
					traj_pair = self.preference_data[idx]
					traj1_states, traj1_actions = traj_pair[0]  # Tuple of (states, actions) arrays
					traj2_states, traj2_actions = traj_pair[1]  # Tuple of (states, actions) arrays
					preference = traj_pair[2]
					
					# Convert states and actions to tensors
					states1 = torch.tensor(traj1_states, dtype=torch.float).to(self.reward_model.device)
					actions1 = np.array(traj1_actions)
					if isinstance(self.env.action_space, gym.spaces.Discrete):
						if actions1.ndim == 1:
							actions1 = actions1.astype(int)
							actions1 = np.eye(self.action_dim)[actions1]
					else:
						if actions1.ndim == 1:
							actions1 = actions1[:, None]  # Ensure shape [T, action_dim]
					actions1 = torch.tensor(actions1, dtype=torch.float32).to(self.reward_model.device)
					
					states2 = torch.tensor(traj2_states, dtype=torch.float).to(self.reward_model.device)
					actions2 = np.array(traj2_actions)
					if isinstance(self.env.action_space, gym.spaces.Discrete):
						if actions2.ndim == 1:
							actions2 = actions2.astype(int)
							actions2 = np.eye(self.action_dim)[actions2]
					else:
						if actions2.ndim == 1:
							actions2 = actions2[:, None]  # Ensure shape [T, action_dim]
					actions2 = torch.tensor(actions2, dtype=torch.float32).to(self.reward_model.device)
					
					# Get rewards for each trajectory
					rewards1 = self.reward_model(states1, actions1).mean()
					rewards2 = self.reward_model(states2, actions2).mean()
					
					# Scale rewards to prevent extreme values
					rewards1 = torch.tanh(rewards1)
					rewards2 = torch.tanh(rewards2)
					
					# Calculate preference loss
					logits = rewards1 - rewards2
					logits = logits.unsqueeze(0)
					target = torch.tensor([preference], dtype=torch.float).to(self.reward_model.device)
					loss = nn.BCEWithLogitsLoss()(logits, target)
					batch_loss += loss
				
				# Average loss over batch
				batch_loss = batch_loss / len(batch_indices)
				batch_loss.backward()
				
				# Gradient clipping
				torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
				
				self.reward_model.optimizer.step()
				total_train_loss += batch_loss.item()
			
			# Validation
			self.reward_model.eval()
			with torch.no_grad():
				for idx in val_indices:
					traj_pair = self.preference_data[idx]
					traj1_states, traj1_actions = traj_pair[0]
					traj2_states, traj2_actions = traj_pair[1]
					preference = traj_pair[2]
					
					# Convert states and actions to tensors
					states1 = torch.tensor(traj1_states, dtype=torch.float).to(self.reward_model.device)
					actions1 = np.array(traj1_actions)
					if isinstance(self.env.action_space, gym.spaces.Discrete):
						if actions1.ndim == 1:
							actions1 = actions1.astype(int)
							actions1 = np.eye(self.action_dim)[actions1]
					else:
						if actions1.ndim == 1:
							actions1 = actions1[:, None]
					actions1 = torch.tensor(actions1, dtype=torch.float32).to(self.reward_model.device)
					
					states2 = torch.tensor(traj2_states, dtype=torch.float).to(self.reward_model.device)
					actions2 = np.array(traj2_actions)
					if isinstance(self.env.action_space, gym.spaces.Discrete):
						if actions2.ndim == 1:
							actions2 = actions2.astype(int)
							actions2 = np.eye(self.action_dim)[actions2]
					else:
						if actions2.ndim == 1:
							actions2 = actions2[:, None]
					actions2 = torch.tensor(actions2, dtype=torch.float32).to(self.reward_model.device)
					
					# Get rewards for each trajectory
					rewards1 = self.reward_model(states1, actions1).mean()
					rewards2 = self.reward_model(states2, actions2).mean()
					
					# Scale rewards
					rewards1 = torch.tanh(rewards1)
					rewards2 = torch.tanh(rewards2)
					
					logits = rewards1 - rewards2
					logits = logits.unsqueeze(0)
					target = torch.tensor([preference], dtype=torch.float).to(self.reward_model.device)
					loss = nn.BCEWithLogitsLoss()(logits, target)
					total_val_loss += loss.item()
			
			# Calculate average losses
			avg_train_loss = total_train_loss / n_batches
			avg_val_loss = total_val_loss / len(val_indices)
			
			# Early stopping
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= patience:
					print(f"Early stopping at epoch {epoch + 1}")
					break			
			if (epoch + 1) % 5 == 0:
				print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
