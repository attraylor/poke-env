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

from poke_env.data import POKEDEX, MOVES, ABILITYDEX, ITEMS, STR_TO_ID
from poke_env.utils import to_id_str
from models import DQN, Embedding_DQN


def fix_type(s):
	return s[0] + s[1:].lower()

class BigBoyRLPlayer(Gen7EnvSinglePlayer):
	def embed_battle(self, battle):
		embedding_indices = {}
		# -1 indicates that the move does not have a base power
		# or is not available
		moves_base_power = -np.ones(4)
		moves_dmg_multiplier = np.ones(4)
		stab = np.zeros(4)
		move_types = np.zeros(4)
		switch_pokemon_length = 3
		max_team_size = 6
		active_pokemon_idx = max_team_size - 1
		switch_features = np.zeros(max_team_size * switch_pokemon_length)



		for pokemon_idx in range(max_team_size):# only five valid switches at a time
			switch_idx = pokemon_idx * switch_pokemon_length
			#Active Pokemon
			if pokemon_idx == active_pokemon_idx:
				current_pokemon = battle.active_pokemon
			elif pokemon_idx >= len(battle._available_switches):
				continue
			else: #i < len(self._available_switches)
				current_pokemon = battle._available_switches[pokemon_idx]
			this_pokemons_indices = {}

			#pokemon species id
			this_pokemons_indices["species_ids"] = STR_TO_ID["species"][to_id_str(current_pokemon._species)]

			#ability id
			this_pokemons_indices["ability_ids"] = STR_TO_ID["abilities"][to_id_str(current_pokemon._ability)]

			#ability id
			if current_pokemon.item is not None:
				try:
					this_pokemons_indices["item_ids"] = STR_TO_ID["items"][to_id_str(current_pokemon._item)] #noitem offset
				except KeyError: #No Item
					this_pokemons_indices["item_ids"] = 0
			#type
			types = [0,0]
			types[0] = STR_TO_ID["types"][fix_type(current_pokemon.type_1.name)]
			if current_pokemon.type_2 is not None:
				types[1] = STR_TO_ID["types"][fix_type(current_pokemon.type_2.name)]
			else:
				types[1] = 0
			this_pokemons_indices["type_ids"] = types

			#hp percentage
			try:
				this_pokemons_indices["hp_percentage"] = current_pokemon._current_hp / current_pokemon._max_hp
			except TypeError:
				print("calculating hp percentage failed for {}".format(current_pokemon._species))
				this_pokemons_indices["hp_percentage"] = 1

			#move ids
			move_ids = [0,0,0,0]
			move_pp = [0,0,0,0]
			for i, move in enumerate(current_pokemon.moves):
				move_ids[i] = STR_TO_ID["moves"][move]
				'''try:
					move_pp[i] = move._current_pp / move._max_pp
				except ZeroDivisionError:
					print("Move division by zero for pp calc", move, move._current_pp, current_pokemon._species)
				'''
			this_pokemons_indices["move_ids"] = move_ids
			this_pokemons_indices["pp"] = move_pp



			#stats
			our_scaled_base_stats = np.zeros(6)
			for i, (name, scaling_factor) in enumerate([
					("hp", 255),
					("atk", 170),
					("def", 230),
					("spa", 170),
					("spd", 230),
					("spe", 160),
					]):
				our_scaled_base_stats[i] = current_pokemon._base_stats[name] / (1.0 * scaling_factor)

			this_pokemons_indices["base_stats"] = our_scaled_base_stats


			embedding_indices[pokemon_idx] = this_pokemons_indices
		return_dir = {}
		final_length = 0
		for key in embedding_indices[active_pokemon_idx].keys():
			if type(embedding_indices[active_pokemon_idx][key]) in [list, np.ndarray]:
				list_size = len(embedding_indices[active_pokemon_idx][key])
				return_dir[key] = np.zeros(6 * list_size)
				final_length += 6 * list_size
				is_list = True
			else:
				return_dir[key] = np.zeros(6)
				is_list = False
				final_length += 6
			for pokemon_idx in range(0, max_team_size):
				if pokemon_idx >= len(battle._available_switches) and not pokemon_idx == max_team_size - 1:
					continue
				if is_list == True:
					list_start = pokemon_idx * list_size
					list_stop = (pokemon_idx + 1) * list_size
					'''print(embedding_indices[pokemon_idx][key])
					print(return_dir[key])
					print(list_start)
					print(list_stop)
					print(list_size)'''
					return_dir[key][list_start:list_stop] = embedding_indices[pokemon_idx][key]
				else:
					return_dir[key][pokemon_idx] = embedding_indices[pokemon_idx][key]
		return return_dir
		final_return = []
		for i in list(["species_id", "ability_id", "item_id", "moves", "type_1", "type_2", "hp_percentage"]):
			final_return.append(return_dir[i])
		final_return = [j for i in final_return for j in i]
		print(final_return, len(final_return))
		return final_return

	def compute_reward(self, battle) -> float:
		return self.reward_computing_helper(
			battle, fainted_value=2, hp_value=1, victory_value=30
		)



class StabSimpleRLPlayer(Gen7EnvSinglePlayer):
	def embed_battle(self, battle):
		# -1 indicates that the move does not have a base power
		# or is not available
		moves_base_power = -np.ones(4)
		moves_dmg_multiplier = np.ones(4)
		stab = np.zeros(4)
		for i, move in enumerate(battle.available_moves):
			moves_base_power[i] = (
				move.base_power / 100
			)  # Simple rescaling to facilitate learning
			if move.type:
				moves_dmg_multiplier[i] = move.type.damage_multiplier(
					battle.opponent_active_pokemon.type_1,
					battle.opponent_active_pokemon.type_2,
				)
				if move.type == battle.active_pokemon.type_1  or move.type == battle.active_pokemon.type_2:
				   stab[i] = 1
		# We count how many pokemons have not fainted in each team
		remaining_mon_team = (
			len([mon for mon in battle.team.values() if mon.fainted]) / 6
		)
		remaining_mon_opponent = (
			len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
		)

		# Final vector with 10 components
		return np.concatenate(
			[
				moves_base_power,
				stab,
				moves_dmg_multiplier,
				[remaining_mon_team, remaining_mon_opponent],
			]
		)

	def compute_reward(self, battle) -> float:
		return self.reward_computing_helper(
			battle, fainted_value=2, hp_value=1, victory_value=30
		)



class SimpleRLPlayer(Gen7EnvSinglePlayer):
	def embed_battle(self, battle):
		# -1 indicates that the move does not have a base power
		# or is not available
		moves_base_power = -np.ones(4)
		moves_dmg_multiplier = np.ones(4)
		for i, move in enumerate(battle.available_moves):
			moves_base_power[i] = (
				move.base_power / 100
			)  # Simple rescaling to facilitate learning
			if move.type:
				moves_dmg_multiplier[i] = move.type.damage_multiplier(
					battle.opponent_active_pokemon.type_1,
					battle.opponent_active_pokemon.type_2,
				)

		# We count how many pokemons have not fainted in each team
		remaining_mon_team = (
			len([mon for mon in battle.team.values() if mon.fainted]) / 6
		)
		remaining_mon_opponent = (
			len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
		)

		# Final vector with 10 components
		return np.concatenate(
			[
				moves_base_power,
				moves_dmg_multiplier,
				[remaining_mon_team, remaining_mon_opponent],
			]
		)

	def compute_reward(self, battle) -> float:
		return self.reward_computing_helper(
			battle, fainted_value=2, hp_value=1, victory_value=30
		)

class MaxDamagePlayer(RandomPlayer):
	def choose_move(self, battle):
		# If the player can attack, it will
		if battle.available_moves:
			# Finds the best move among available ones
			best_move = max(battle.available_moves, key=lambda move: move.base_power)
			return self.create_order(best_move)

		# If no attack is available, a random switch will be made
		else:
			return self.choose_random_move(battle)
