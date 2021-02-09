import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from tqdm import trange
from copy import deepcopy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# import torchvision.transforms as T

from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.player import Player


from sklearn.decomposition import PCA #Grab PCA functions
import matplotlib.pyplot as plt

from poke_env.data import STR_TO_ID, ID_TO_STR, MOVES
from poke_env.utils import to_id_str
import relevant_conditions

def homogenize_vectors(vectors):
	tensors = []
	for vector in vectors:
		tensor = torch.FloatTensor(vector)
		if len(tensor.shape) == 1: #Batch size is 1:
			tensor = tensor.unsqueeze(0)
		elif len(tensor.shape) == 3: #Batch size is 1:
			tensor = tensor.squeeze(0)
		tensors.append(tensor)
	return tensors

class TeenyBoy_DQN(nn.Module):
	def __init__(self, config):

		super(TeenyBoy_DQN, self).__init__()
		#Embedding dimension sizes
		self.input_dim = 4
		self.hidden_dim = config.complete_state_hidden_dim
		self.output_dim = 22
		self.layers = []
		self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
		self.layer2 = nn.Linear(self.hidden_dim, self.output_dim)
		self.layer2.weight.data.fill_(0)
		self.layer2.bias.data.fill_(0)
		#self.layers.append(nn.Linear(self.input_dim,config.hidden_dim))
		#for i in range(1, config.num_layers):
		#	self.layers.append(nn.Linear(config.hidden_dim,config.hidden_dim))
		#self.layers.append(nn.Linear(self.hidden_dim,config.output_dim))


	def forward(self, state_dict):
		"""State representation right now:
			- team: List of pokemon object dictionaries, len = team_size
				- Pokemon: Dict of {id_field : value},
					-Value: is one of:
						-list
					 	-int ("ids" in id_field name): for an embedding index
						-float: between 0 and 1, scalar value
						-bool: for 0/1 input
			- opponent_team: List of pokemon object dictionaries
			"""
		batch_size = len(state_dict["weather"])
		active_pokemon = state_dict["team"][0]
		features = torch.FloatTensor(active_pokemon["move_powers"])
		state_embedding = self.layer2(F.relu(self.layer1(features)))
		'''move_powers = np.zeros(4)
		moves_dmg_multiplier = np.zeros(4)
		team_health = np.zeros(2)
		active_pokemon = state_dict["team"][0]
		moves = active_pokemon["move_ids"]
		for idx, move_idx in moves:
			move_name = ID_TO_STR[move_idx]
			move_power = MOVES[move_name]["basePower"]
			move_power = move_power * 1.0 / 150
			move_powers[idx] = move_power
			move_type = STR_TO_ID[MOVES[move_name]["type"]]
			opponent_types = state_dict["opponent_team"][0]["type_ids"]

			moves_dmg_multiplier

		x = complete_state_concatenation
		for layer in self.complete_state_linear_layers[:-1]:
			x = F.relu(layer(x))
		state_embedding = self.complete_state_linear_layers[-1](x)'''

		#TODO (longterm): move residuals
		return state_embedding
