import gym
import math
import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from itertools import count
from tqdm import trange
from copy import deepcopy
import os
import wandb
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from PIL import Image
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from teams import teams

from config_utils import create_config

import argparse

# import torchvision.transforms as T

print("file", __file__)
import sys
sys.path += ["/gpfs/data/epavlick/atraylor/Pokemon/poke-env/src"]


print(sys.path)

from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.player import Player
from poke_env.player.baselines import RandomPlayer, SimpleHeuristicsPlayer

from io import StringIO


from sklearn.decomposition import PCA #Grab PCA functions
import matplotlib.pyplot as plt

from poke_env.data import POKEDEX, MOVES
from poke_env.utils import to_id_str
from bigboy_model_1layer import *
from teenyboy_model import *
from players import *


from config_utils import create_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))


def stack_dicts(objects):
	#Objects is List<Dict>
	if type(objects) == list and type(objects[0]) == tuple:
		objects = objects[0]

	final_dict = {}
	for key in objects[0].keys():
		if len(objects[0][key].shape) == 1:
			needs_flatten = False
			vector_shape = objects[0][key].shape[0]
		else:
			needs_flatten = True
			vector_shape = objects[0][key].shape[0] * objects[0][key].shape[1]
		arr = np.zeros((len(objects), vector_shape))
		for i in range(0, len(objects)):
			if needs_flatten == True:
				arr[i] = objects[i][key].reshape(vector_shape)
			else:
				arr[i] = objects[i][key]
		final_dict[key] = arr
	#TODO: Convert dict of embed_state dictionary objects to one dictionary
	#That is correctly batched.
	return final_dict


class ReplayMemory(torch.utils.data.Dataset):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def __getitem__(self, idx):
		return self.memory[idx]

	def __len__(self):
		return len(self.memory)

def custom_bigboy_collate(batch):
	"""
	Collates a list of transitions into a minibatch interpretable by our embedder.
	Input:
		batch: List of Transition tuples (s, a, ns, r)
			s: Dictionary of string: Dict / string: List (embed_battle output)
			a: int
			ns: Dictionary of string: Dict / string: List (embed_battle output)
			r: float
	Output:
		state_batch: Dictionary of string: Dict / string: List (embed_battle output)
			WHERE: Every element of dict is batch_size by its original value
		a: List of ints
		ns: Dictionary of string: Dict / string: List (embed_battle output)
			WHERE: Every element of dict is batch_size by its original value
		r: List of floats
	"""
	#Transition = namedtuple('Transition',
	#						('state', 'action', 'next_state', 'reward'))
	#reward_batch = Torch.LongTensor([reward for (_, _, _, reward) in batch])
	action_batch = []
	reward_batch = []
	state_batch = {}
	state_batch["team"] = [defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list), defaultdict(list), defaultdict(list)]
	state_batch["opponent_team"] = [defaultdict(list)]
	state_batch["weather"] = []
	state_batch["fields"] = []
	state_batch["player_side_conditions"] = []
	state_batch["opponent_side_conditions"] = []

	next_state_batch = {}
	next_state_batch["team"] = [defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list), defaultdict(list), defaultdict(list)]
	next_state_batch["opponent_team"] = [defaultdict(list)]
	next_state_batch["weather"] = []
	next_state_batch["fields"] = []
	next_state_batch["player_side_conditions"] = []
	next_state_batch["opponent_side_conditions"] = []
	for (state, action, next_state, reward) in batch:
		for key in state.keys():
			if key in ["team", "opponent_team"]:
				for pokemon_idx, pokemon in enumerate(state[key]):
					for key2 in pokemon.keys():
						state_batch[key][pokemon_idx][key2].append(pokemon[key2])
			else:
				state_batch[key].append(state[key])

		assert next_state is not None
		for key in next_state.keys():
			if key in ["team", "opponent_team"]:
				for pokemon_idx, pokemon in enumerate(next_state[key]):
					for key2 in pokemon.keys():
						next_state_batch[key][pokemon_idx][key2].append(pokemon[key2])
			else:
				next_state_batch[key].append(next_state[key])
			#
		action_batch.append(action)
		reward_batch.append(reward)

	action_batch = torch.LongTensor(action_batch)
	reward_batch = torch.FloatTensor(reward_batch)
	return state_batch, action_batch, next_state_batch, reward_batch

def fit(player, nb_steps):
	global loss_hist
	push = 0
	global config
	global reward_hist
	episode_durations = []
	#self.reset_states() #TODO: Impl
	#tq = trange(nb_steps, desc="Reward: 0")
	tq = range(nb_steps)
	episode_reward = 0
	current_step_number = 0
	stopped_adding = False
	for i_episode in tq:
		state = None
		if state is None:  # start of a new episode
			# Initialize the environment and state
			state = deepcopy(env_player.reset())
			if type(state) in [list, np.ndarray]:
				state = torch.autograd.Variable(torch.Tensor(state), requires_grad=False)
			for t in count():
				# Select and perform an action

				action = select_action(state, env_player.gen8_legal_action_mask(env_player._current_battle),
						test=False, eps_start = config.eps_start, eps_end = config.eps_end,
						eps_decay = config.eps_decay,
						nb_episodes = config.nb_training_steps, current_step = i_episode)
				next_state, reward, done, info = env_player.step(action.item())
				#next_state = deepcopy(torch.autograd.Variable(torch.Tensor(next_state), requires_grad=False))
				reward = torch.FloatTensor([reward])
				episode_reward += reward
				#tq.set_description("Reward: {:.3f}".format(episode_reward.item()))
				reward_hist.append(episode_reward.item())
				#x = input("reward {} ".format(reward))
				#if i_episode < 10:
				#if push < 2 and reward != 0:
				#	print("PUSH REWARD", reward)
				memory.push(state, action, next_state, reward)
				#	push += 1
				#elif stopped_adding == False:
				#	print("loss stopped!!!!!!!!!!!!!!")
				#	stopped_adding = True
				# Move to the next state
				state = next_state

				# Perform one step of the optimization (on the policy network)
				current_step_number += 1
				if current_step_number % config.optimize_every == 0:
					optimize_model()
				if done:
					episode_durations.append(t + 1)
					#print('Episode {}: reward: {:.3f}'.format(i_episode, reward))
					#plot_durations()
					break
			# Update the target network, copying all weights and biases in DQN

			if i_episode % config.target_update == 0:
				#print("***updating target network*******************")
				target_net.load_state_dict(policy_net.state_dict())

	print("avg battle length: {}".format(sum(episode_durations) / len(episode_durations)))


torch.set_printoptions(sci_mode=False)

def test(player, nb_episodes):
	#tq = trange(nb_episodes, desc="Reward: 0")
	tq = range(nb_episodes)
	episode_reward = 0
	for i_episode in tq:
		state = None
		if state is None:  # start of a new episode
			# Initialize the environment and state
			state = deepcopy(env_player.reset())
			if type(state) in [list, np.ndarray]:
				state = torch.autograd.Variable(torch.Tensor(state), requires_grad=False)
			for t in count():
				# Select and perform an action
				action = select_action(state, env_player.gen8_legal_action_mask(env_player._current_battle),
						test=True)
				next_state, reward, done, info = env_player.step(action.item())
				#next_state = deepcopy(torch.autograd.Variable(torch.Tensor(next_state), requires_grad=False))
				reward = torch.FloatTensor([reward])
				episode_reward += reward
				#tq.set_description("Reward: {:.3f}".format(episode_reward.item()))
				# Store the transition in memory
				memory.push(state, action, next_state, reward)
				# Move to the next state
				state = next_state

				if done:
					#plot_durations()
					break

def select_action(state, action_mask = None, test= False, eps_start = 0.9,
		eps_end = 0.05, eps_decay = 200, nb_episodes = 2000, current_step = 0):
	#Epsilon greedy action selection with action mask from environment
	verbose = False
	with torch.no_grad():
		q_values = policy_net(state,verbose=verbose)
	q_values = q_values.squeeze(0)

	assert len(q_values.shape) == 1
	nb_actions = q_values.shape[0]
	if test == False:
		current_eps = eps_end + (eps_start - eps_end) * \
			np.exp(-1 * current_step / eps_decay)

	if action_mask != None:
		#Mask out to only actions that are legal within the state space.
		#action_mask_neg_infinity = [float("-inf") if action_mask[i] == 0 else 1 for i in range(0, len(action_mask))]
		action_mask_neg_infinity = [-1000000 if action_mask[i] == 0 else 0 for i in range(0, len(action_mask))]
		action_mask_neg_infinity = torch.autograd.Variable(torch.FloatTensor(action_mask_neg_infinity), requires_grad=False)
		legal_actions = [i for i in range(0, len(action_mask)) if action_mask[i] == 1]#np.where(action_mask == 1)

		if len(legal_actions) == 0:
			print("no actions legal! unclear why this happens-- potentially trapped and disabled? Maybe bug?", action_mask)
			return torch.LongTensor([[0]])
		if test == False and np.random.uniform() < current_eps:
			action = np.random.choice(legal_actions)#np.random.randint(0, nb_actions)
		else:
			action = np.argmax(q_values + action_mask_neg_infinity)
			if test == True:
				#print(q_values)
				#print(q_values + action_mask_neg_infinity)
				#print(action)
				#x = input("x")
				pass
	else: #This shouldnt be called
		if test == False and np.random.uniform() < current_eps:
			action = np.random.randint(0, nb_actions)
		else:
			action = np.argmax(q_values)
			print("\n\n\nhmmmm\n\n\n")
	return torch.LongTensor([[action]])



def optimize_model():
	global loss_hist
	global config
	'''if len(memory) < config.batch_size:
		return'''


	'''transitions = memory.sample(config.batch_size)
	# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	# detailed explanation). This converts batch-array of Transitions
	# to Transition of batch-arrays.
	batch = Transition(*zip(*transitions))'''
	train_data = torch.utils.data.DataLoader(memory, batch_size = config.batch_size, collate_fn = custom_bigboy_collate)
	batch_cap = config.batch_cap
	batch_loss = 0
	for idx, batch in enumerate(train_data):
		'''print(json.dumps(batch[0], indent=1))
		x = input("batch print s")
		print(json.dumps(batch[2], indent=1))
		x = input("batch print s'")'''

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		state_batch, action_batch, next_state, reward_batch = batch
		#non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,next_state)), dtype=torch.bool)
		#non_final_next_states = torch.stack([s for s in next_state
		#											if s is not None])
		#non_final_next_states = stack_dicts([s for s in batch.next_state
		#											if s is not None])

		'''state_batch = torch.stack(batch.state)#stack_dicts([batch.state])
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)
		'''
		q_values = policy_net(state_batch)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		if config.batch_size == 1:
			q_values = q_values.unsqueeze(1)
		else:
			state_action_values = q_values.gather(1, action_batch.unsqueeze(1))
			#state_action_values torch.FloatTensor([q_values[i][action_batch[i]] for i in range(q_values.shape[0])])
		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(config.batch_size)
		next_state_values = target_net(next_state).max(1)[0].detach()
		#next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * config.gamma) + reward_batch
		# Compute Huber loss
		#print("state_action_values\n")
		actions = action_batch.float().unsqueeze(1)
		diff = state_action_values - expected_state_action_values.unsqueeze(1)
		#print(torch.cat([diff, actions, state_action_values, expected_state_action_values.unsqueeze(1)],dim=1))
		'''print("next_state_values", next_state_values)
		print("reward batch", reward_batch)
		print("expected_state_action_values", expected_state_action_values)'''
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
		#x = input("sav")
		loss_hist.append(loss)
		batch_loss += loss
		# Optimize the model
		optimizer.zero_grad()
		loss.backward()
		for name, param in policy_net.named_parameters():
			#print(name)
			param.grad.data.clamp_(-1, 1)
			#print(param.grad.data)
		optimizer.step()
		if idx > batch_cap:
			break
	return




def dqn_training(player, nb_steps):
	fit(player, nb_steps=nb_steps)
	player.complete_current_battle()

def dqn_evaluation(player, nb_episodes):
	# Reset battle statistics
	player.reset_battles()
	test(player, nb_episodes=nb_episodes)

	print(
		"DQN Evaluation: %d victories out of %d episodes"
		% (player.n_won_battles, nb_episodes)
	)

from poke_env.teambuilder.teambuilder import Teambuilder

class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.teams = [self.join_team(self.parse_showdown_team(team)) for team in teams]

    def yield_team(self):
        return np.random.choice(self.teams)


if __name__ == "__main__":
	global config
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="filepath")
	parser.add_argument("--saved_model", type=str, default="filepath")
	args = parser.parse_args()
	hyperparameter_defaults = create_config(args.config)
	hyperparameter_defaults["saved_model"] = args.saved_model
	wandb.init(config=hyperparameter_defaults)
	config = wandb.config

	file_time = str(time.time())

	custom_builder = RandomTeamFromPool([teams[config.our_team_name]])
	custom_builder2 = RandomTeamFromPool([teams[config.opponent_team_name]])

	env_player = BigBoyRLPlayer(
		player_configuration=PlayerConfiguration("SimpleRLPlayer", None),
		battle_format="gen8ou",
		team=custom_builder,
		server_configuration=LocalhostServerConfiguration,
	)

	opponent = RandomPlayer(
		player_configuration=PlayerConfiguration("Random player", None),
		battle_format="gen8ou",
		team=custom_builder2,
		server_configuration=LocalhostServerConfiguration,
	)



	second_opponent = MaxDamagePlayer(
		player_configuration=PlayerConfiguration("Max damage player", None),
		battle_format="gen8ou",
		team=custom_builder2,
		server_configuration=LocalhostServerConfiguration,
	)

	third_opponent = SimpleHeuristicsPlayer(
		player_configuration=PlayerConfiguration("Simple heuristic player", None),
		battle_format="gen8ou",
		team=custom_builder2,
		server_configuration=LocalhostServerConfiguration,
	)

	n_actions = len(env_player.action_space)


	policy_net = TeenyBoy_DQN(config)
	target_net = TeenyBoy_DQN(config)
	policy_net.load_state_dict(torch.load(config.saved_model))
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	#optimizer = optim.RMSprop(policy_net.parameters())
	optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
	memory = ReplayMemory(config.memory_size)

	steps_done = 0

	loss_hist = []
	reward_hist = []


	old_stdout = sys.stdout
	result = StringIO()
	sys.stdout = result#open("results/"+file_time+"/log_games.txt","w+")

	env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=opponent,
		env_algorithm_kwargs={"nb_episodes": config.nb_evaluation_episodes},
	)

	"""env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=second_opponent,
		env_algorithm_kwargs={"nb_episodes": config.nb_evaluation_episodes},
	)

	env_player.play_against(
		env_algorithm=dqn_evaluation,
		opponent=third_opponent,
		env_algorithm_kwargs={"nb_episodes": config.nb_evaluation_episodes},
	)"""

	sys.stdout = old_stdout

	result_string = result.getvalue()
	winrates = result_string.split("\n")
	random_winrate = float(winrates[0].split(" ")[2])/config.nb_evaluation_episodes
	'''max_winrate = float(winrates[1].split(" ")[2])/config.nb_evaluation_episodes
	heuristic_winrate = float(winrates[2].split(" ")[2])/config.nb_evaluation_episodes

	wandb.log({"random_winrate": random_winrate, "max_winrate": max_winrate, "heuristic_winrate": heuristic_winrate})
	print(random_winrate, max_winrate, heuristic_winrate)'''
	print(random_winrate)
	print('Complete')

	#env.render()
	env_player.close()
