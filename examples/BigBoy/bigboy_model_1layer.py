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

from poke_env.data import STR_TO_ID
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

class BigBoy_DQN1L(nn.Module):
	def __init__(self, config):

		super(BigBoy_DQN1L, self).__init__()
		#Embedding dimension sizes
		self.species_emb_dim = config.species_emb_dim
		self.move_emb_dim = config.move_emb_dim
		self.item_emb_dim = config.item_emb_dim
		self.ability_emb_dim = config.ability_emb_dim
		self.type_emb_dim = config.type_emb_dim
		self.status_emb_dim = config.status_emb_dim
		self.weather_emb_dim = config.weather_emb_dim

		self.species_embedding = nn.Embedding(len(STR_TO_ID["species"]) + 1, self.species_emb_dim)
		self.move_embedding = nn.Embedding(len(STR_TO_ID["moves"]) + 1, self.move_emb_dim)
		self.item_embedding = nn.Embedding(len(STR_TO_ID["items"]) + 1, self.item_emb_dim)
		self.ability_embedding = nn.Embedding(len(STR_TO_ID["abilities"]) + 1, self.ability_emb_dim)
		self.type_embedding = nn.Embedding(len(STR_TO_ID["types"]) + 1, self.type_emb_dim)
		#Todo: magic number from players.py
		self.status_embedding = nn.Embedding(len(relevant_conditions.statuses), self.status_emb_dim)
		self.weather_embedding = nn.Embedding(len(relevant_conditions.weathers), self.weather_emb_dim)

		#Pokemon_embedder
		pokemon_emb_input_dim = self.species_emb_dim + 4 * config.move_encoder_hidden_dim + self.item_emb_dim + self.ability_emb_dim + 2 * self.type_emb_dim + self.status_emb_dim + 1 + 6 + 7 + len(relevant_conditions.volatiles) #TODO: + scalars
		self.pokemon_emb_linear_layer = nn.Linear(pokemon_emb_input_dim, config.pokemon_embedding_hidden_dim)

		#team_embedding network
		team_embedding_input_dim = 6 * config.pokemon_embedding_hidden_dim
		self.team_embedding_linear_layer = nn.Linear(team_embedding_input_dim, config.team_embedding_hidden_dim)

		#move_encoder_input_size = move emb dim + type emb dim + power (1) + P/S/O (3) + Accuracy (1)
		move_encoder_input_dim = (self.move_emb_dim) + (self.type_emb_dim) + 3
		self.move_encoder_linear_layer = nn.Linear(move_encoder_input_dim,config.move_encoder_hidden_dim)

		#opponent_team network
		#Magic numbers: 1 (hp %), 6 (base stats), 7 (boosts)
		opponent_input_dim = self.species_emb_dim + 2 * self.type_emb_dim + self.status_emb_dim + 1 + 6 + 7 + len(relevant_conditions.volatiles)
		self.opponent_linear_layer = nn.Linear(opponent_input_dim, config.opponent_hidden_dim)

		complete_state_input_dim = config.team_embedding_hidden_dim + config.opponent_hidden_dim + self.weather_emb_dim + 2 * len(relevant_conditions.side_conditions) + len(relevant_conditions.fields)
		self.complete_state_linear_layer = nn.Linear(complete_state_input_dim, config.complete_state_output_dim)
		'''new_weights = torch.FloatTensor(np.zeros(self.complete_state_linear_layers[-1].weight.shape))
		new_bias = torch.FloatTensor(np.zeros(self.complete_state_linear_layers[-1].bias.shape))
		self.complete_state_linear_layers[-1].bias.data.fill_(.1)
		self.complete_state_linear_layers[-1].weight.data.fill_(.1)'''

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
		pokemon_vectors = []
		for idx, pokemon in enumerate(state_dict["team"]):

			#MOVE ENCODER

			#Input: Batch x 4
			#Output: Batch x 4 x move_emb_dim
			#pokemon = state_dict["team"][pokemon_name]
			move_vectors = self.move_embedding(torch.LongTensor(pokemon["move_ids"]))

			#Input: Batch x 4
			#Output: Batch x 4 x type_emb_dim
			move_type_vector = self.type_embedding(torch.LongTensor(pokemon["move_type_ids"]))

			vectors = [
					move_vectors, #B x 4 x M_E_D
					move_type_vector, #B x 4 x T_E_D
					pokemon["move_powers"], #B x 4 (Unsqueeze below)
					pokemon["move_accuracies"], #B x 4 (Unsqueeze below)
					pokemon["moves_physical_or_special"] #B x 4 (Unsqueeze below)
					]
			vectors = [torch.FloatTensor(vector) for vector in vectors]
			vectors[2] = vectors[2].unsqueeze(-1)
			vectors[3] = vectors[3].unsqueeze(-1)
			vectors[4] = vectors[4].unsqueeze(-1)
			if len(vectors[0].shape) == 2: #Batch size is 1: #TODO doesnt cover for 1dims
				vectors = [vector.unsqueeze(0) for vector in vectors]
			x = torch.cat(vectors, dim = 2) #B x 4 x (M_E_D) + (T_E_D) + 3
			x = F.relu(self.move_encoder_linear_layer(x))

			#Final output size: B X 4 X self.move_encoder_output_size
			move_encoder_output = x
			move_encoder_output = move_encoder_output.reshape(move_encoder_output.shape[0], move_encoder_output.shape[1] * move_encoder_output.shape[2])

			#STOP MOVE ENCODER: REST OF EMBEDDINGS

			#Input: Batch x 1
			#Output: Batch x species_emb_dim
			pokemon_id = torch.LongTensor([pokemon["species_id"]])
			species_vector = self.species_embedding(pokemon_id)
			#TODO Unsqueeze

			#Input: Batch x 1
			#Output: Batch x item_emb_dim
			item_vector = self.item_embedding(torch.LongTensor([pokemon["item_id"]]))
			#TODO Unsqueeze

			#Input: Batch x 1
			#Output: Batch x ability_emb_dim
			ability_vector = self.ability_embedding(torch.LongTensor([pokemon["ability_id"]]))
			#TODO Unsqueeze

			#Input: Batch x 2 (2 from pokemon types)
			#Output: Batch x 2 x type_emb_dim
			#TODO: Concat pokemon["move_type_ids"] and pokemon["type_ids"]
			type_vector = self.type_embedding(torch.LongTensor(pokemon["type_ids"]))
			if len(type_vector.shape) == 2: #We only passed in one example
				type_vector = type_vector.reshape(type_vector.shape[0] * type_vector.shape[1])
				type_vector = type_vector.unsqueeze(0)
			elif len(type_vector.shape) == 3: #We only passed in one example
				type_vector = type_vector.reshape(type_vector.shape[0], type_vector.shape[1] * type_vector.shape[2])

			#Input: Batch x 1
			#Output: Batch x status_emb_dim
			status_vector = self.status_embedding(torch.LongTensor([pokemon["status_id"]]))
			#TODO Unsqueeze

			#TODO: Concatenate scalars
			vectors = [
				move_encoder_output, #B X (MEOSx4) will be reshaped earlier
				species_vector, #B x S_E_D
				item_vector, #B x I_E_D
				ability_vector, #B x A_E_D
				type_vector, #B x T_E_D
				status_vector, #B x St_E_D
				pokemon["hp_percentage"], #B x 1
				pokemon["stats"], #B x 6
				pokemon["boosts"], #B x 8
				pokemon["volatiles"], #B x len(relevant_conditions.volatiles)
					]

			pokemon_tensors = homogenize_vectors(vectors)
			'''for vector in vectors:
				tensor = torch.FloatTensor(vector)
				if len(tensor.shape) == 1: #Batch size is 1:
					tensor = tensor.unsqueeze(0)
				pokemon_tensors.append(tensor)'''


			x = torch.cat(pokemon_tensors, dim=1)
			x = F.relu(self.pokemon_emb_linear_layer(x))
			pokemon_representation = x
			pokemon_vectors.append(pokemon_representation)

		#Pokemon_vectors: Size 6x pokemon_embedder_hidden_dim
		x = torch.cat(pokemon_vectors, dim=1)
		x =  F.relu(self.team_embedding_linear_layer(x))
		team_embedding =  x

		#opponent_team_embedding
		opponent_team = []
		for idx, opponent_pokemon in enumerate(state_dict["opponent_team"]):
			#opponent_pokemon = state_dict["opponent_team"][opponent_pokemon_name]
			species_vector = self.species_embedding(torch.LongTensor([opponent_pokemon["species_id"]]))
			type_vector = self.type_embedding(torch.LongTensor(opponent_pokemon["type_ids"]))
			if len(type_vector.shape) == 2: #We only passed in one example
				type_vector = type_vector.reshape(type_vector.shape[0] * type_vector.shape[1])
				type_vector = type_vector.unsqueeze(0)
			elif len(type_vector.shape) == 3: #We only passed in one example
				type_vector = type_vector.reshape(type_vector.shape[0], type_vector.shape[1] * type_vector.shape[2])

			status_vector = self.status_embedding(torch.LongTensor([opponent_pokemon["status_id"]]))

			vectors = [
						species_vector, #B x S_E_D
						type_vector, #B x T_E_D
						status_vector, #B x St_E_D
						opponent_pokemon["hp_percentage"], #B x 1
						opponent_pokemon["base_stats"], #B x 6
						opponent_pokemon["boosts"], #B x 7
						opponent_pokemon["volatiles"], #B x len(relevant_conditions.volatiles)
						]


			opponent_tensors = homogenize_vectors(vectors)

			concatenated_input_for_pokemon_encoder = torch.cat(opponent_tensors, dim=1)
			x = concatenated_input_for_pokemon_encoder
			#TODO (longterm): Fix this after we know more about opponent's team
		x =  F.relu(self.opponent_linear_layer(x))
		opponent_embedding =  x


		weather_embedding = self.weather_embedding(torch.LongTensor(state_dict["weather"]))

		if len(weather_embedding.shape) == 3:
			if weather_embedding.shape[2] == 1:
				weather_embedding = weather_embedding.squeeze(2)
			elif weather_embedding.shape[1] == 1:
				weather_embedding = weather_embedding.squeeze(1)


		vectors = [
			team_embedding,
			opponent_embedding,
			weather_embedding,
			state_dict["player_side_conditions"],
			state_dict["opponent_side_conditions"],
			state_dict["fields"]
		]
		vectors = homogenize_vectors(vectors)
		try:
			complete_state_concatenation = torch.cat(vectors, dim=1)
		except RuntimeError:
			print([vector.shape for vector in vectors])
			print(self.weather_embedding(torch.LongTensor(state_dict["weather"])).shape)
			sys.exit(1)

		x = complete_state_concatenation
		x = F.relu(self.complete_state_linear_layer(x))
		state_embedding = x
		#TODO (longterm): move residuals
		return state_embedding



class Embedding_DQN(nn.Module):
	def __init__(self, input_shape=10, pokemon_emb_dim = 32, move_emb_dim = 32, hidden_dim=128, hidden_dim2=64, output_shape=22):
		self.pokemon_emb_dim = pokemon_emb_dim
		self.move_emb_dim = move_emb_dim
		super(DQN, self).__init__()
		self.pokemon_embedding = nn.Embedding(1000, pokemon_emb_dim)
		self.move_embedding = nn.Embedding(1000, move_emb_dim)
		first_dim = 2 * pokemon_emb_dim + 4 * move_emb_dim + input_shape - 6
		self.lin1 = nn.Linear(first_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim2)
		self.lin3 = nn.Linear(hidden_dim2, output_shape)

	def forward(self, x):
		if x.dim() == 1:
			d = 0
			pokemon_ids = x[0:2]
			move_ids = x[2:6]
			numeric_input = x[6:]
			our_active_embs = self.pokemon_embedding(pokemon_ids[0].long())
			opponent_active_embs = self.pokemon_embedding(pokemon_ids[1].long())
			move_embs = self.move_embedding(move_ids.long())
			move_embs = move_embs.reshape(move_embs.shape[0]* move_embs.shape[1])
		else:
			d = 1
			pokemon_ids = x[:,0]
			opp_pokemon_ids = x[:,1]
			move_ids = x[:,2:6]
			numeric_input = x[:,6:]
			our_active_embs = self.pokemon_embedding(pokemon_ids.long())
			opponent_active_embs = self.pokemon_embedding(opp_pokemon_ids.long())
			move_embs = self.move_embedding(move_ids.long())

			#TODO: is this stable?
			move_embs = move_embs.reshape(move_embs.shape[0], move_embs.shape[1]* move_embs.shape[2])
		if x.shape[-1] > 6:
			x = torch.cat([our_active_embs, opponent_active_embs, move_embs, numeric_input], dim=d)
		else:
			x = torch.cat([our_active_embs, opponent_active_embs,  move_embs], dim=d)
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		return x

class DQN(nn.Module):
	def __init__(self, input_shape=10,hidden_dim=128, hidden_dim2=64, output_shape=22):
		super(DQN, self).__init__()
		self.lin1 = nn.Linear(input_shape, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim2)
		self.lin3 = nn.Linear(hidden_dim2, output_shape)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		return x
