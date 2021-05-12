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

POKEMONID_ONEHOTS = [[0, 0, 0, 0,0,0]] * 1037
POKEMONID_ONEHOTS[1036] = [1, 0, 0, 0, 0, 0]
POKEMONID_ONEHOTS[388]  = [0, 1, 0, 0, 0, 0]
POKEMONID_ONEHOTS[984]  = [0, 0, 1, 0, 0, 0]
POKEMONID_ONEHOTS[152]  = [0, 0, 0, 1, 0, 0]
POKEMONID_ONEHOTS[439]  = [0, 0, 0, 0, 1, 0]
POKEMONID_ONEHOTS[365]  = [0, 0, 0, 0, 0, 1]


class SinglelineMediumBoy_DQN(nn.Module):
	def __init__(self, config):

		super(SinglelineMediumBoy_DQN, self).__init__()
		#Embedding dimension sizes

		self.use_pokemon_encoder = True

		if self.use_pokemon_encoder == True:
			self.pokemon_encoder_size = 16
			self.pokemon_encoder_input_size = 19 + 7 + 1
			self.pokemon_encoder = []
			self.pokemon_encoder.append(nn.Linear(self.pokemon_encoder_input_size, self.pokemon_encoder_size))#self.pokemon_encoder_size)
			self.pokemon_encoder.append(nn.ReLU())
			self.pokemon_encoder.append(nn.Linear(self.pokemon_encoder_size, self.pokemon_encoder_input_size))#self.pokemon_encoder_size)
			self.pokemon_encoder = nn.Sequential(*self.pokemon_encoder)


		self.type_embedding = nn.Embedding(19, 19)
		self.type_embedding.weight.data = torch.FloatTensor(np.eye(19))
		self.type_embedding.weight.requires_grad = False

		self.status_embedding = nn.Embedding(7, 7)
		self.status_embedding.weight.data = torch.FloatTensor(np.eye(7))
		self.status_embedding.weight.requires_grad = False

		self.input_dim = 288
		self.hidden_dim = config.complete_state_hidden_dim
		self.output_dim = 22
		self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
		layers = [nn.LayerNorm(self.input_dim), self.input_layer]
		layers.append(nn.ReLU())
		for i in range(0, config.num_layers):
			layers.append(nn.LayerNorm(self.hidden_dim))
			layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
			layers.append(nn.ReLU())
		self.layers = nn.Sequential(*layers)
		self.last_layer = nn.Linear(self.hidden_dim, self.output_dim)
		self.last_layer.weight.data.fill_(0)
		self.last_layer.bias.data.fill_(0)
		self.pokemonid_onehots = nn.Embedding(1037, 6)
		self.pokemonid_onehots.weight.data = torch.FloatTensor(POKEMONID_ONEHOTS)
		self.pokemonid_onehots.weight.requires_grad = False
		#self.layers.append(nn.Linear(self.input_dim,config.hidden_dim))
		#for i in range(1, config.num_layers):
		#	self.layers.append(nn.Linear(config.hidden_dim,config.hidden_dim))
		#self.layers.append(nn.Linear(self.hidden_dim,config.output_dim))

	def get_input_size(self):
		input_size = 0
		argument_enabled_to_input_size = {
			"our_pokemon_1_move_powers": 4,
			"our_pokemon_1_move_type_ids": 76,
			"our_pokemon_1_hp_percentage": 1,
			"our_pokemon_2_6_hp_percentage": 5,
			"our_pokemon_1_boosts": 7,
			"our_pokemon_1_type_ids": 19,
			"our_pokemon_2_6_type_ids": 95,
			"opponent_pokemon_active_type_ids": 19,
			"opponent_pokemon_active_boosts": 7,
			"opponent_pokemon_active_hp_percentage": 1,
			"our_pokemon_1_volatiles": 1, #TODO: More volatiles
			"opponent_pokemon_active_volatiles": 1,
			"our_side_conditions": 1,
			"opponent_side_conditions": 1,
			"our_pokemon_1_status_id": 1,
			"our_pokemon_2_6_status_id": 5,
			"opponent_pokemon_active_status_id": 1
			}
		for key,value in self.teambuilding_config.items() and value == True:
			if key in argument_enabled_to_input_size.keys():
				input_size += argument_enabled_to_input_size[key]
		return input_size

	def forward(self, batch, field_to_idx, verbose=False):
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
		if len(batch.shape) == 1:
			batch_size = 1
			batch = batch.unsqueeze(0)
		else:
			batch_size = batch.shape[0]
		features = []
		features.append(torch.FloatTensor(batch[:,field_to_idx["our_pokemon_1_move_powers"]]))
		move_type_ids = self.type_embedding(batch[:,field_to_idx["our_pokemon_1_move_type_ids"]].long())
		features.append(move_type_ids.reshape(batch_size, move_type_ids.shape[1] * move_type_ids.shape[2]))
		features.append(torch.FloatTensor(batch[:,field_to_idx["our_pokemon_1_hp_percentage"]]))

		features.append(torch.FloatTensor(batch[:,field_to_idx["our_pokemon_1_boosts"]]))

		for i in range(1, 7):
			pokemon_object = []
			pokemon_object.append(self.type_embedding(batch[:,field_to_idx["our_pokemon_{}_type_ids".format(i)][0]].long()) + self.type_embedding(batch[:,field_to_idx["our_pokemon_{}_type_ids".format(i)][1]].long()))
			pokemon_object.append(torch.FloatTensor(batch[:,field_to_idx["our_pokemon_{}_hp_percentage".format(i)]]))
			pokemon_object.append(self.status_embedding(batch[:,field_to_idx["our_pokemon_{}_status_id".format(i)]].long()))
			pokemon_object = torch.cat(pokemon_object,dim=1)
			if self.use_pokemon_encoder == True:
				features.append(self.pokemon_encoder(pokemon_object))
			else:
				features.append(pokemon_object)


		features.append(self.type_embedding(batch[:,field_to_idx["opponent_pokemon_active_type_ids"][0]].long()) + self.type_embedding(batch[:,field_to_idx["opponent_pokemon_active_type_ids"][1]].long()))
		features.append(torch.FloatTensor(batch[:,field_to_idx["opponent_pokemon_active_boosts"]]))
		features.append(torch.FloatTensor(batch[:,field_to_idx["opponent_pokemon_active_hp_percentage"]]))

		#TAUNTED?
		features.append(torch.FloatTensor(batch[:,field_to_idx["our_pokemon_1_volatiles"][-3]]).unsqueeze(1))
		features.append(torch.FloatTensor(batch[:,field_to_idx["opponent_pokemon_active_volatiles"][-3]]).unsqueeze(1))

		#ROCKS?
		features.append(torch.FloatTensor(batch[:,field_to_idx["our_side_conditions"][7]]).unsqueeze(1))
		features.append(torch.FloatTensor(batch[:,field_to_idx["opponent_side_conditions"][7]]).unsqueeze(1))

		#STATUSED?
		features.append(self.status_embedding(batch[:,field_to_idx["opponent_pokemon_active_status_id"]].long()))

		#TODO: knock off, move IDs, opponent team backup pokemon, trapped?, uturn

		features = torch.cat(features,dim=1)

		if verbose == True:
			print("")
			print(features)

		state_embedding = self.last_layer(self.layers(features))

		#TODO (longterm): move residuals
		return state_embedding
