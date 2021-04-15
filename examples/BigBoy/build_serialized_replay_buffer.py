import gym
import math
import random
import numpy as np
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
import jsonlines


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from teams import *

import argparse

# import torchvision.transforms as T

print("file", __file__)
import sys
print(sys.path)

from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.player import Player
from poke_env.player.baselines import RandomPlayer, SimpleHeuristicsPlayer


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
	global config
	global reward_hist
	episode_durations = []
	#self.reset_states() #TODO: Impl
	tq = trange(nb_steps, desc="Reward: 0")
	episode_reward = 0
	current_step_number = 0
	stopped_adding = False
	for i_episode in tq:
		state = None
		if state is None:  # start of a new episode
			# Initialize the environment and state
			state = deepcopy(env_player.reset())
			#if type(state) in [list, np.ndarray]:
			#	state = torch.autograd.Variable(torch.Tensor(state), requires_grad=False)
			for t in count():
				# Select and perform an action

				action = env_player.choose_move(env_player._current_battle)

				next_state, reward, done, info = env_player.step(action.item())
				#next_state = deepcopy(torch.autograd.Variable(torch.Tensor(next_state), requires_grad=False))
				reward = torch.FloatTensor([reward])
				episode_reward += reward
				tq.set_description("Reward: {:.3f}".format(episode_reward.item()))
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
				if memory.position == memory.capacity:
					return
				if done:
					episode_durations.append(t + 1)
					#print('Episode {}: reward: {:.3f}'.format(i_episode, reward))
					#plot_durations()
					break
			# Update the target network, copying all weights and biases in DQN

	print("avg battle length: {}".format(sum(episode_durations) / len(episode_durations)))

def get_transitions_for_replay_memory(player, nb_steps):
	fit(player, nb_steps=nb_steps)
	player.complete_current_battle()

from poke_env.teambuilder.teambuilder import Teambuilder

class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.teams = [self.join_team(self.parse_showdown_team(team)) for team in teams]

    def yield_team(self):
        return np.random.choice(self.teams)

def serialize_memory(memory, writepath):
	with jsonlines.open(writepath, mode='w') as writer:
    	writer.write(memory)


if __name__ == "__main__":
	global config
	hyperparameter_defaults = dict(
		experiment_name = "BigBoy",
		memory_size = 10000, #How many S,A,S',R transitions we keep in memory
	)

	wandb.init(config=hyperparameter_defaults)
	config = wandb.config

	writepath = "ReplayBuffers/{}.jsonl".format(config.experiment_name)
	if not os.path.exists(writepath):
		os.makedirs(writepath)



	custom_builder = RandomTeamFromPool([team_starters])
	custom_builder2 = RandomTeamFromPool([team_starters])

	player_one = MaxDamagePlayer(
		player_configuration=PlayerConfiguration("Max player one", None),
		battle_format="gen8ou",
		team=custom_builder,
		server_configuration=LocalhostServerConfiguration,
	)

	player_two = MaxDamagePlayer(
		player_configuration=PlayerConfiguration("Max player two", None),
		battle_format="gen8ou",
		team=custom_builder2,
		server_configuration=LocalhostServerConfiguration,
	)

	memory = ReplayMemory(config.memory_size)


	player_one.play_against(
		env_algorithm=get_transitions_for_replay_memory,
		opponent=player_two,
		env_algorithm_kwargs={"nb_steps": 100000000},
	)
	env_player.close()

	serialize_memory(memory, writepath)
