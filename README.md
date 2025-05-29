# RLHF_EGGang
Reinforcement Learning with Human Feedback - a group coding project at EPFL
The preference dataset usually comes from surveys, but here, it is synthetically generated from trained policies (using a more traditionnal RL algorithm). PPO-RLHF (learning a reward model from a preference dataset, then performing PPO) and DPO (which does not require learning a reward model) were implemented in standard RL environments : Cartpole and Mountaincar from OpenAI's Gymnasium library.

# Code organisation

Folders:
 - `PPO_RLHF`: scripts to run PPO and to train a reward model
 - `figures`: store some figures for the poster/report
 - `saved_data`: save trajectory data (mostly in CSV format)
 - `saved_policies`: temporary checkpoint pytorch models, and a few of them that were saved to get $\pi_{1,2,3}$
 - `utils`: scripts to train policies before generating a synthetic preference dataset
Files:
 - `cartpole.ipynb`: pipeline to generate the preference dataset for Cartpole
 - `eval_DPO.ipynb`: pipeline to implement DPO in Cartpole and Mountaincar
 - `eval_ppo-rlhf.ipynb`: pipeline to implement PPO-RLHF in Cartpole and Mountaincar


# Requirements

See `requirements.txt` for the necessary packages and versions. Some issues may be due to the updates in gym / gymnasium.

Code was run on Python 3.13

# Members - Epsilon-greedy Gang
 - Jérôme Barras
 - Marc Bonhôte
 - Noé Prat
 - Mikael Schär
 - Antoine Violet
 