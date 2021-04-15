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

FWG_ONEHOTS = {0: [0, 0, 0], 7: [1, 0, 0], 10: [0, 1, 0], 18: [0, 0, 1]}

POKEMONID_ONEHOTS = {
	1036: [1, 0, 0, 0, 0, 0],
	388:  [0, 1, 0, 0, 0, 0],
	984:  [0, 0, 1, 0, 0, 0],
	152:  [0, 0, 0, 1, 0, 0],
	439:  [0, 0, 0, 0, 1, 0],
	365:  [0, 0, 0, 0, 0, 1],
}



class MediumBoy_DQN(nn.Module):

	def __init__(self, config):

		super(MediumBoy_DQN, self).__init__()
		#Embedding dimension sizes
		self.type_embedding_style = "twohot"
		self.type_embedding = nn.Embedding(18, 18)
		self.type_embedding.weight.data = torch.FloatTensor(np.eye(18))
		self.type_embedding.weight.requires_grad = False
		self.input_dim = 18
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

	def build_model(self):
		self.input_dim = 0

		#How will the model represent typing information?
		if config.represent_types_as in ["onehot", "twohot"]:
			self.type_embedding = nn.Embedding(18, 18)
			self.type_embedding.weight.data = torch.FloatTensor(np.eye(18))
			self.type_embedding.weight.requires_grad = False
			if config.represent_types_as == "onehot":
				self.type_embedding_size = 18 * 2
			else:
				self.type_embedding_size = 18
		else:
			print("Typing rep not implemented")
			sys.exit(1)
			pass

		if config.include_our_pokemon_species_typing == True:
			self.input_dim += self.type_embedding_size * config.number_pokemon
		if config.include_opponent_pokemon_species_typing == True:
			self.input_dim += self.type_embedding_size #TODO: Make this a little more modular.

		#How does the model represent species information?
		if config.represent_species_as in ["onehot"]:
			self.num_species = 6
			self.species_embedding = nn.Embedding(self.num_species, self.num_species)
			self.type_embedding.weight.data = torch.FloatTensor(np.eye(6))
			self.type_embedding.weight.requires_grad = False
			self.species_embedding_size = self.num_species
		else:
			print("Species rep not implemented")
			sys.exit(1)
			pass

		if config.include_our_pokemon_species_embedding == True:
			self.input_dim += self.species_embedding_size * config.number_pokemon
		if config.include_opponent_pokemon_species_embedding == True:
			self.input_dim += self.species_embedding_size #TODO: Make this a little more modular.

		#How does the model represent health?
		if config.include_our_pokemon_health == True:
			self.input_dim += config.number_pokemon #Todo: make more modular (just active?)
		if config.include_opponent_pokemon_health == True:
			self.input_dim += 1 #Todo: make more modular (all pokemon?)

		#How does the model represent move information?
		if config.include_our_pokemon_move_power == True:
			self.input_dim += 4 #Todo: represent back pokemon move power as well?
		if config.include_our_pokemon_move_typing == True:
			if config.represent_move_typing_as == "same_as_species":
				self.move_typing_emb_dim = 18 #Hardcoded
			else:
				print("move typing emb not implemented")
				sys.exit(1)
			self.input_dim += 4 * self.move_typing_emb_dim




		#Make our model
		self.hidden_dim = config.complete_state_hidden_dim
		self.output_dim = 22
		self.layers = []
		self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
		self.layer2 = nn.Linear(self.hidden_dim, self.output_dim)
		self.layer2.weight.data.fill_(0)
		self.layer2.bias.data.fill_(0)
		return

	def get_features(self, state_dict, config):
		batch_size = len(state_dict["weather"])

		input_features = []
		if config.include_our_pokemon_species_typing == True:
			species_typing = self.type_embedding(state_dict["team"][:]["type_ids"])
			if config.represent_species_as == "twohot":
				species_typing = species_typing[:, :, 0, :] + species_typing[:, :, 1, :] #[batch_size, team_size (6), type_0, type_emb_dim]
			input_features.append(species_typing)
		if config.include_our_pokemon_species_typing == True:
			opponent_species_typing = self.type_embedding(state_dict["opponent_team"][0]["type_ids"]) #TODO: modularity
			if config.represent_species_as == "twohot":
				opponent_species_typing = opponent_species_typing[:, 0, :] + opponent_species_typing[:, 1, :] #[batch_size, team_size (6), type_0, type_emb_dim]
			input_features.append(opponent_species_typing)

		if config.include_our_pokemon_species_embedding == True:
			input_features.append(self.species_embedding(state_dict["team"][:]["species_id"]))
		if config.include_opponent_species_embedding == True:
			input_features.append(self.species_embedding(state_dict["opponent_team"][0]["species_id"])) #TODO: modularity

		if len(batch_size) == 1:
			features = torch.cat(input_features)
		else:
			features = torch.cat(input_features,dim=1))
		assert len(features) = self.input_dim
		active_pokemon = state_dict["team"][0]
		backup_pokemon1 = state_dict["team"][1]
		backup_pokemon2 = state_dict["team"][2]
		move_features = torch.FloatTensor(active_pokemon["move_powers"])
		opponent_pokemon = state_dict["opponent_team"][0]
		opp_health = torch.FloatTensor(state_dict["opponent_team"][0]["hp_percentage"])
		health = torch.FloatTensor(active_pokemon["hp_percentage"])
		if len(move_features.shape) == 1:
			active_pokemon_type_ids = torch.FloatTensor(FWG_ONEHOTS[active_pokemon["type_ids"][0]])
			backup_pokemon1_type_ids = torch.FloatTensor(FWG_ONEHOTS[backup_pokemon1["type_ids"][0]])
			backup_pokemon2_type_ids = torch.FloatTensor(FWG_ONEHOTS[backup_pokemon2["type_ids"][0]])

			#type_ids = torch.LongTensor(active_pokemon["move_type_ids"])
			opponent_type_ids = torch.FloatTensor(FWG_ONEHOTS[opponent_pokemon["type_ids"][0]])
			features = torch.cat([move_features, active_pokemon_type_ids, backup_pokemon1_type_ids, backup_pokemon2_type_ids, opponent_type_ids, health, opp_health])

		else:
			active_pokemon_type_ids = torch.FloatTensor([FWG_ONEHOTS[x[0]] for x in active_pokemon["type_ids"]])
			backup_pokemon1_type_ids = torch.FloatTensor([FWG_ONEHOTS[x[0]] for x in backup_pokemon1["type_ids"]])
			backup_pokemon2_type_ids = torch.FloatTensor([FWG_ONEHOTS[x[0]] for x in backup_pokemon2["type_ids"]])
			opponent_type_ids = torch.FloatTensor([FWG_ONEHOTS[x[0]] for x in opponent_pokemon["type_ids"]])
			feats = [move_features, active_pokemon_type_ids, backup_pokemon1_type_ids, backup_pokemon2_type_ids, opponent_type_ids]
			'''for feature in feats:
				print(feature.shape)'''

			features = torch.cat([move_features, active_pokemon_type_ids, backup_pokemon1_type_ids, backup_pokemon2_type_ids, opponent_type_ids,  health, opp_health],dim=1)
		if verbose == True:
			print("")
			print(features)
		return features

	def forward(self, state_dict, verbose=False):
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
		features = self.get_features(state_dict, self.config)


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
