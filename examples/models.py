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

from poke_env.data import POKEDEX, MOVES
from poke_env.utils import to_id_str


class BigBoy_DQN(nn.Module):
	def __init__(self, input_shape=62, pokemon_emb_dim = 32, move_emb_dim = 32, hidden_dim=128, hidden_dim2=64, output_shape=18):
		self.pokemon_emb_dim = pokemon_emb_dim
		self.move_emb_dim = move_emb_dim
		self.input_shape = input_shape
		self.type_emb_dim = 5
		super(BigBoy_DQN, self).__init__()
		self.pokemon_embedding = nn.Embedding(1000, pokemon_emb_dim)
		self.move_embedding = nn.Embedding(1000, move_emb_dim)
		self.type_embedding = nn.Embedding(19, self.type_emb_dim)
		#first_dim = input_shape - 16 + self.type_emb_dim * 16
		first_dim = 6 * (pokemon_emb_dim + 4 * move_emb_dim + 2 * self.type_emb_dim)
		self.lin1 = nn.Linear(first_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim2)
		self.lin3 = nn.Linear(hidden_dim2, hidden_dim2)
		self.lin4 = nn.Linear(hidden_dim2, hidden_dim2)
		self.lin5 = nn.Linear(hidden_dim2, output_shape)

	def forward(self, state_value_dict):
		#Each value of state_value_dict is of dim [batch, team_size * length]
		#pokemon_embeddings: [batch, team_size]
		#move_embeddings: [batch, team_size * 4]
		pokemon_vectors = self.pokemon_embedding(torch.LongTensor(state_value_dict["species_ids"]))
		move_vectors = self.move_embedding(torch.LongTensor(state_value_dict["move_ids"]))
		type_vectors = self.type_embedding(torch.LongTensor(state_value_dict["type_ids"]))
		x = []
		for vec in [pokemon_vectors, move_vectors, type_vectors]:
			x.append(torch.flatten(vec))
		x = torch.cat(x, 0)
		print(x.shape)
		#print(pokemon_vectors.shape, move_vectors.shape, type_vectors.shape)
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		x = F.relu(self.lin4(x))
		x = self.lin5(x)
		return x



class Embedding_DQN(nn.Module):
	def __init__(self, input_shape=10, pokemon_emb_dim = 32, move_emb_dim = 32, hidden_dim=128, hidden_dim2=64, output_shape=18):
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
	def __init__(self, input_shape=10,hidden_dim=128, hidden_dim2=64, output_shape=18):
		super(DQN, self).__init__()
		self.lin1 = nn.Linear(input_shape, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim2)
		self.lin3 = nn.Linear(hidden_dim2, output_shape)

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		return x
