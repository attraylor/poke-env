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
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.player import Player


from sklearn.decomposition import PCA #Grab PCA functions
import matplotlib.pyplot as plt

from poke_env.data import STR_TO_ID
from poke_env.utils import to_id_str
from bigboy_model import DQN, Embedding_DQN
from poke_env.environment.status import Status

import relevant_conditions

def fix_type(s):
	return s[0] + s[1:].lower()



class BigBoyRLPlayer(Gen8EnvSinglePlayer):
	def embed_battle(self, battle):
		final_state_dict = {}
		team = []
		active_pokemon = battle.active_pokemon
		switches = battle.available_switches
		fainted_padding = [None] * (5 - len(switches))
		pokemon_objects = [active_pokemon] + switches + fainted_padding
		assert len(pokemon_objects) == 6
		for idx, pokemon in enumerate(pokemon_objects):#battle.team.keys(): #thought this was a method
			#pokemon = battle.team[pokemon_key]
			pokemon_information = {}

			#TODO: Bump the rest of these declarations to the top.
			species_id = 0
			types = [0,0]
			move_ids = [0,0,0,0]
			move_pp = [0,0,0,0]
			move_type_ids = [0,0,0,0]
			move_powers = [0,0,0,0]
			move_accuracies = [0,0,0,0]
			move_physical_or_special = [0,0,0,0]
			item_id = 0
			hp_percentage = [0]
			ability_id = 0
			status_id = 0
			scaled_stats = [0] * 6
			boosts = [0] * 7
			volatiles = [0] * len(relevant_conditions.volatiles)

			if not pokemon == None and not pokemon.fainted and not pokemon._species == "zoroark":
				# -1 indicates that the move does not have a base power
				# or is not available

				species_id = STR_TO_ID["species"][to_id_str(pokemon._species)]


				#Type ids

				types[0] = STR_TO_ID["types"][fix_type(pokemon.type_1.name)]
				if pokemon.type_2 is not None:
					types[1] = STR_TO_ID["types"][fix_type(pokemon.type_2.name)]
				else:
					pass

				#Move ids
				#move pp
				#move type
				#move power
				#move accuracy
				#physical or special or nondamaging


				for i, move in enumerate(pokemon.moves):
					if i == 4:
						break
					move_obj = pokemon.moves[move]
					#TODO: Sometimes this throws an error. Why?
					try:
						move_ids[i] = STR_TO_ID["moves"][move]
					except IndexError:
						print(pokemon)
						print(pokemon.moves)
						print(move_ids, move_obj)
						print("weirdo move error", move,  i)
						print(STR_TO_ID["moves"][move])
					try:
						move_pp[i] = move_obj._current_pp / move_obj.max_pp
					except ZeroDivisionError:
						print("Move division by zero for pp calc", move, move_obj._current_pp, pokemon._species)
					move_accuracies[i] = move_obj.accuracy
					move_type_ids[i] = STR_TO_ID["types"][fix_type(move_obj.type.name)]
					move_powers[i] = move_obj.base_power / 150.0 #TODO: Explosion?
					if move_obj.category.name == "PHYSICAL":
						move_physical_or_special[i] = 0
					elif move_obj.category.name == "SPECIAL":
						move_physical_or_special[i] = 1
					elif move_obj.category.name == "STATUS":
						move_physical_or_special[i] = 2
					else:
						print("woops")
						sys.exit(1)



				#Item id

				if pokemon.item is not None:
					try:
						item_id = STR_TO_ID["items"][to_id_str(pokemon._item)] #noitem offset
					except KeyError: #TODO: Why does this happen?
						pass


				#Ability id
				ability_id = STR_TO_ID["abilities"][to_id_str(pokemon._ability)]
				#status
				if pokemon.status == Status.FNT:
					print("You shouldn't be in here")
					sys.exit(1)
				else:
					status_id = relevant_conditions.statuses.index(pokemon.status)

				#HP (%perentage)

				try:
					hp_percentage[0] = pokemon._current_hp * 1.0 / pokemon._max_hp
				except TypeError:
					print("calculating hp percentage failed for {}".format(pokemon._species))




				for i, name in enumerate(["hp", "atk", "def", "spa", "spd", "spe"]):
					if name == "hp":
						stat = pokemon._max_hp
						stat_max = 669
						stat_min = 15
						#Min HP at level 50: 15
						#Max HP at level 100: 669
					else:
						stat = pokemon.stats[name]
						stat_max = 714
						stat_min = 70
						#Min other stat at level 100: 70
						#Max other stat at level 100: 714
					try:
						scaled_stats[i] = (stat - stat_min) / (1.0 * stat_max - stat_min)
					except TypeError:
						print("TypeError: Zoroark")
						scaled_stats[i] = 0



				'''#TODO (longterm): NATURES

				#Base stats
				scaled_base_stats = [0] * 6
				for i, (name, scaling_factor) in enumerate([
						("hp", 255),
						("atk", 170),
						("def", 230),
						("spa", 170),
						("spd", 230),
						("spe", 160),
						]):
					scaled_base_stats[i] = pokemon._base_stats[name] / (1.0 * scaling_factor)

				pokemon_information["base_stats"] = scaled_base_stats


				#IVs
				ivs = [0] * 6
				print(pokemon.stats)
				for i, name in enumerate(["hp","atk", "def","spa","spd","spe"]):
					ivs[i] = pokemon.ivs[name] / 32.0
				pokemon_information["ivs"] = ivs

				#EVs
				#Total EVs: 510
				#Max per stat: 252
				#Can only increment by 4 at a time
				#Number of increments = 252 / 4
				evs = [0] * 6
				for i, name in enumerate(["hp","atk", "def","spa","spd","spe"]):
					evs[i] = pokemon.evs[name] / 252.0
				pokemon_information["evs"] = evs'''

				#boosts
				#For each stat "hp","atk", "def","spa","spd","spe" AND "eva", "acc"
				#Multipliers for stats
				#13 levels [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 ]
				#Acc: [-3, -2, -1, 0, 1, 2, 3]

				#TODO (longterm): What is the best encoding of these values? Scalar? 0 to 1? -1 to 1?
				#TODO: Change this for opponent too

				for i, name in enumerate(["atk", "def","spa","spd","spe", "evasion"]):
					boosts[i] = (pokemon.boosts[name] + 6) / 12.0
				boosts[-1] = (pokemon.boosts["accuracy"] + 3) / 6.0


				#volatiles

				active_effects = pokemon.effects
				volatiles = [1 if x in active_effects else 0 for x in relevant_conditions.volatiles]

			pokemon_information["species_id"] = species_id
			pokemon_information["type_ids"] = types
			pokemon_information["move_ids"] = move_ids
			pokemon_information["pp"] = move_pp
			pokemon_information["move_type_ids"] = move_type_ids
			pokemon_information["move_powers"] = move_powers
			pokemon_information["move_accuracies"] = move_accuracies
			pokemon_information["move_physical_or_special"] = move_physical_or_special

			pokemon_information["item_id"] = item_id
			pokemon_information["ability_id"] = ability_id
			pokemon_information["status_id"] = status_id

			pokemon_information["stats"] = scaled_stats
			pokemon_information["hp_percentage"] = hp_percentage

			pokemon_information["boosts"] = boosts
			pokemon_information["volatiles"] = volatiles

			team.append(pokemon_information)
			#isActive

		side_conditions = [0] * len(relevant_conditions.side_conditions)
		active_side_conditions = battle.side_conditions
		our_side_conditions = [1 if x in active_side_conditions else 0 for x in relevant_conditions.side_conditions ]


		final_state_dict["team"] = team
		final_state_dict["player_side_conditions"] = our_side_conditions

		opponent_team = []
		opponent_pokemon_information = {}
		opponent_pokemon = battle.opponent_active_pokemon


		opponent_types = [0,0]
		opponent_species_id = 0
		opponent_hp_percentage = [1]
		opponent_status_id = 0
		opponent_scaled_base_stats = [0] * 6
		opponent_boosts = [0] * 7
		opponent_volatiles = [0] * len(relevant_conditions.volatiles)

		#opponent active pokemon id
		if battle.opponent_active_pokemon.fainted == False:
			opponent_species_id = STR_TO_ID["species"][to_id_str(opponent_pokemon._species)]

			#opponent pokemon type ids

			opponent_types[0] = STR_TO_ID["types"][fix_type(opponent_pokemon.type_1.name)]
			if opponent_pokemon.type_2 is not None:
				opponent_types[1] = STR_TO_ID["types"][fix_type(opponent_pokemon.type_2.name)]
			else:
				pass


			#Opponent HP (%expected perentage)
			#TODO: Does this code actually get the expected percentage based on the opponent's health bar?

			try:
				opponent_hp_percentage[0] = opponent_pokemon._current_hp * 1.0 / opponent_pokemon._max_hp
			except TypeError:
				print("calculating hp percentage failed for {}".format(opponent_pokemon._species))


			#opponent status
			opponent_status_id = relevant_conditions.statuses.index(opponent_pokemon.status)

			#opponent base stats
			for i, (name, scaling_factor) in enumerate([
					("hp", 255),
					("atk", 170),
					("def", 230),
					("spa", 170),
					("spd", 230),
					("spe", 160),
					]):
				opponent_scaled_base_stats[i] = opponent_pokemon._base_stats[name] / (1.0 * scaling_factor)

			#opponent boost


			for i, name in enumerate(["atk", "def","spa","spd","spe", "evasion"]):
				opponent_boosts[i] = (opponent_pokemon.boosts[name] + 6) / 12.0
			opponent_boosts[-1] = (opponent_pokemon.boosts["accuracy"] + 3) / 6.0



			#opponent volatiles
			active_effects = opponent_pokemon.effects
			opponent_volatiles = [1 if x in active_effects else 0 for x in relevant_conditions.volatiles]


		#TODO (longterm):
		#Keeping track of opponent's moves
		#Keeping track of opponent's item
		#Keeping track of opponent's ability
		#Possible abilities

		opponent_pokemon_information["species_id"] = opponent_species_id
		opponent_pokemon_information["type_ids"] = opponent_types
		opponent_pokemon_information["hp_percentage"] = opponent_hp_percentage
		opponent_pokemon_information["boosts"] = opponent_boosts
		opponent_pokemon_information["base_stats"] = opponent_scaled_base_stats
		opponent_pokemon_information["volatiles"] = opponent_volatiles
		opponent_pokemon_information["status_id"] = opponent_status_id

		opponent_team.append(opponent_pokemon_information)

		opponent_side_conditions = [0] * len(relevant_conditions.side_conditions)
		opponent_active_side_conditions = battle.opponent_side_conditions
		opponent_side_conditions = [1 if x in opponent_active_side_conditions else 0 for x in relevant_conditions.side_conditions]

		final_state_dict["opponent_team"] = opponent_team
		final_state_dict["opponent_side_conditions"] = opponent_side_conditions


		weather = relevant_conditions.weathers.index(battle.weather)
		final_state_dict["weather"] = [weather]


		fields = [0] * len(relevant_conditions.fields)
		active_fields = battle.fields
		fields = [1 if x in active_fields else 0 for x in relevant_conditions.fields ]
		final_state_dict["fields"] = fields
		return final_state_dict

	def compute_reward(self, battle) -> float:
		return self.reward_computing_helper(
			battle, fainted_value=5, hp_value=0, victory_value=10
		)


class SingleLineRLPlayer(Gen8EnvSinglePlayer):

	def create_pokemon_mapping(self, team):
		"""
		PRECONDITION: Species clause.
		Input:
			team: dict(Pokemon objects)
		Output:
			stable_team: {str pokemon_name : int idx}
			stable_team_inv: {int idx : str_pokemon_name}
		"""
		stable_team = {}
		stable_team_inv = {}
		stable_team_p1name = {}
		for idx, pokemon_name in enumerate(team.keys()): #str
			species_name = team[pokemon_name].species
			stable_team[species_name] = idx
			stable_team_p1name[species_name] = pokemon_name
			stable_team_inv[idx] = species_name
		self.stable_team = stable_team
		self.stable_team_inv = stable_team_inv
		self.stable_team_p1name = stable_team_p1name

	def embed_battle(self, battle):
		if battle.turn == 1:
			self.create_pokemon_mapping(battle.team)
		state = []
		ind_dict = {}
		index = 0

		active_pokemon = battle.active_pokemon
		switches = battle.available_switches
		fainted_padding = [None] * (5 - len(switches))
		pokemon_objects = [active_pokemon] + switches + fainted_padding

		#active_pokemon = battle.active_pokemon
		#switches = battle.available_switches
		#fainted_padding = [None] * (5 - len(switches))
		pokemon_objects = [battle.active_pokemon] + team
		assert len(pokemon_objects) == 6 + 1
		team_order = ["active", 0, 1, 2, 3, 4, 5]
		#for idx, pokemon in enumerate(pokemon_objects):#battle.team.keys(): #thought this was a method

		for idx, pokemon in zip(team_order, pokemon_objects):
			pokemon_number = idx
			#pokemon = battle.team[pokemon_key]
			pokemon_information = {}

			#TODO: Bump the rest of these declarations to the top.
			species_id = 0
			types = [0,0]
			move_ids = [0,0,0,0]
			move_pp = [0,0,0,0]
			move_type_ids = [0,0,0,0]
			move_powers = [0,0,0,0]
			move_accuracies = [0,0,0,0]
			move_physical_or_special = [0,0,0,0]
			item_id = 0
			hp_percentage = [0]
			ability_id = 0
			status_id = 0
			scaled_stats = [0] * 6
			boosts = [0] * 7
			volatiles = [0] * len(relevant_conditions.volatiles)

			if not pokemon == None and not pokemon.fainted and not pokemon._species == "zoroark":
				# -1 indicates that the move does not have a base power
				# or is not available

				species_id = STR_TO_ID["species"][to_id_str(pokemon._species)]


				#Type ids

				types[0] = STR_TO_ID["types"][fix_type(pokemon.type_1.name)]
				if pokemon.type_2 is not None:
					types[1] = STR_TO_ID["types"][fix_type(pokemon.type_2.name)]
				else:
					pass

				#Move ids
				#move pp
				#move type
				#move power
				#move accuracy
				#physical or special or nondamaging


				for i, move in enumerate(pokemon.moves):
					if i == 4:
						break
					move_obj = pokemon.moves[move]
					#TODO: Sometimes this throws an error. Why?
					try:
						move_ids[i] = STR_TO_ID["moves"][move]
					except IndexError:
						print(pokemon)
						print(pokemon.moves)
						print(move_ids, move_obj)
						print("weirdo move error", move,  i)
						print(STR_TO_ID["moves"][move])
					try:
						move_pp[i] = move_obj._current_pp / move_obj.max_pp
					except ZeroDivisionError:
						print("Move division by zero for pp calc", move, move_obj._current_pp, pokemon._species)
					move_accuracies[i] = move_obj.accuracy
					move_type_ids[i] = STR_TO_ID["types"][fix_type(move_obj.type.name)]
					move_powers[i] = move_obj.base_power / 150.0 #TODO: Explosion?
					if move_obj.category.name == "PHYSICAL":
						move_physical_or_special[i] = 0
					elif move_obj.category.name == "SPECIAL":
						move_physical_or_special[i] = 1
					elif move_obj.category.name == "STATUS":
						move_physical_or_special[i] = 2
					else:
						print("woops")
						sys.exit(1)



				#Item id

				if pokemon.item is not None:
					try:
						item_id = STR_TO_ID["items"][to_id_str(pokemon._item)] #noitem offset
					except KeyError: #TODO: Why does this happen?
						pass


				#Ability id
				ability_id = STR_TO_ID["abilities"][to_id_str(pokemon._ability)]
				#status
				if pokemon.status == Status.FNT:
					print("You shouldn't be in here")
					sys.exit(1)
				else:
					status_id = relevant_conditions.statuses.index(pokemon.status)

				#HP (%perentage)

				try:
					hp_percentage[0] = pokemon._current_hp * 1.0 / pokemon._max_hp
				except TypeError:
					print("calculating hp percentage failed for {}".format(pokemon._species))




				for i, name in enumerate(["hp", "atk", "def", "spa", "spd", "spe"]):
					if name == "hp":
						stat = pokemon._max_hp
						stat_max = 669
						stat_min = 15
						#Min HP at level 50: 15
						#Max HP at level 100: 669
					else:
						stat = pokemon.stats[name]
						stat_max = 714
						stat_min = 70
						#Min other stat at level 100: 70
						#Max other stat at level 100: 714
					try:
						scaled_stats[i] = (stat - stat_min) / (1.0 * stat_max - stat_min)
					except TypeError:
						print("TypeError: Zoroark")
						scaled_stats[i] = 0



				'''#TODO (longterm): NATURES

				#Base stats
				scaled_base_stats = [0] * 6
				for i, (name, scaling_factor) in enumerate([
						("hp", 255),
						("atk", 170),
						("def", 230),
						("spa", 170),
						("spd", 230),
						("spe", 160),
						]):
					scaled_base_stats[i] = pokemon._base_stats[name] / (1.0 * scaling_factor)

				pokemon_information["base_stats"] = scaled_base_stats


				#IVs
				ivs = [0] * 6
				print(pokemon.stats)
				for i, name in enumerate(["hp","atk", "def","spa","spd","spe"]):
					ivs[i] = pokemon.ivs[name] / 32.0
				pokemon_information["ivs"] = ivs

				#EVs
				#Total EVs: 510
				#Max per stat: 252
				#Can only increment by 4 at a time
				#Number of increments = 252 / 4
				evs = [0] * 6
				for i, name in enumerate(["hp","atk", "def","spa","spd","spe"]):
					evs[i] = pokemon.evs[name] / 252.0
				pokemon_information["evs"] = evs'''

				#boosts
				#For each stat "hp","atk", "def","spa","spd","spe" AND "eva", "acc"
				#Multipliers for stats
				#13 levels [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 ]
				#Acc: [-3, -2, -1, 0, 1, 2, 3]

				#TODO (longterm): What is the best encoding of these values? Scalar? 0 to 1? -1 to 1?
				#TODO: Change this for opponent too

				for i, name in enumerate(["atk", "def","spa","spd","spe", "evasion"]):
					boosts[i] = (pokemon.boosts[name] + 6) / 12.0
				boosts[-1] = (pokemon.boosts["accuracy"] + 3) / 6.0


				#volatiles

				active_effects = pokemon.effects
				volatiles = [1 if x in active_effects else 0 for x in relevant_conditions.volatiles]

			for (n, var) in [
				("species_id", species_id),
				("type_ids", types),
				("move_ids", move_ids),
				("move_pp", move_pp),
				("move_type_ids", move_type_ids),
				("move_powers", move_powers),
				("move_accuracies", move_accuracies),
				("move_physical_or_special", move_physical_or_special),
				("item_id", item_id),
				("ability_id", ability_id),
				("status_id", status_id),
				("stats", scaled_stats),
				("hp_percentage", hp_percentage),
				("boosts", boosts),
				("volatiles", volatiles),
				]:
				if type(var) == list:
					state += var
					ind_dict["our_pokemon_{}_{}".format(pokemon_number, n)] = list(range(index, index + len(var)))
					index += len(var)
				elif type(var) in [int, float]:
					state.append(var)
					ind_dict["our_pokemon_{}_{}".format(pokemon_number, n)] = index
					index += 1
				else:
					print("???")
					print(n, var)
					sys.exit(1)

		side_conditions = [0] * len(relevant_conditions.side_conditions)
		active_side_conditions = battle.side_conditions
		our_side_conditions = [1 if x in active_side_conditions else 0 for x in relevant_conditions.side_conditions ]
		state += our_side_conditions
		ind_dict["our_side_conditions"] = list(range(index, index + len(our_side_conditions)))
		index += len(our_side_conditions)

		opponent_pokemon = battle.opponent_active_pokemon


		opponent_types = [0,0]
		opponent_species_id = 0
		opponent_hp_percentage = [1]
		opponent_status_id = 0
		opponent_scaled_base_stats = [0] * 6
		opponent_boosts = [0] * 7
		opponent_volatiles = [0] * len(relevant_conditions.volatiles)

		#opponent active pokemon id
		if battle.opponent_active_pokemon.fainted == False:
			opponent_species_id = STR_TO_ID["species"][to_id_str(opponent_pokemon._species)]

			#opponent pokemon type ids

			opponent_types[0] = STR_TO_ID["types"][fix_type(opponent_pokemon.type_1.name)]
			if opponent_pokemon.type_2 is not None:
				opponent_types[1] = STR_TO_ID["types"][fix_type(opponent_pokemon.type_2.name)]
			else:
				pass


			#Opponent HP (%expected perentage)
			#TODO: Does this code actually get the expected percentage based on the opponent's health bar?

			try:
				opponent_hp_percentage[0] = opponent_pokemon._current_hp * 1.0 / opponent_pokemon._max_hp
			except TypeError:
				print("calculating hp percentage failed for {}".format(opponent_pokemon._species))


			#opponent status
			opponent_status_id = relevant_conditions.statuses.index(opponent_pokemon.status)

			#opponent base stats
			for i, (name, scaling_factor) in enumerate([
					("hp", 255),
					("atk", 170),
					("def", 230),
					("spa", 170),
					("spd", 230),
					("spe", 160),
					]):
				opponent_scaled_base_stats[i] = opponent_pokemon._base_stats[name] / (1.0 * scaling_factor)

			#opponent boost


			for i, name in enumerate(["atk", "def","spa","spd","spe", "evasion"]):
				opponent_boosts[i] = (opponent_pokemon.boosts[name] + 6) / 12.0
			opponent_boosts[-1] = (opponent_pokemon.boosts["accuracy"] + 3) / 6.0



			#opponent volatiles
			active_effects = opponent_pokemon.effects
			opponent_volatiles = [1 if x in active_effects else 0 for x in relevant_conditions.volatiles]


		#TODO (longterm):
		#Keeping track of opponent's moves
		#Keeping track of opponent's item
		#Keeping track of opponent's ability
		#Possible abilities
		for (n, var) in [
			("species_id", opponent_species_id),
			("type_ids", opponent_types),
			#("move_ids", move_ids),
			#("move_pp", move_pp),
			#("move_type_ids", move_type_ids),
			#("move_powers", move_powers),
			#("move_accuracies", move_accuracies),
			#("move_physical_or_special", physical_or_special),
			#("item_id", item_id),
			#("ability_id", ability_id),
			("status_id", opponent_status_id),
			#("stats", scaled_stats),
			("base_stats", opponent_scaled_base_stats),
			("hp_percentage", opponent_hp_percentage),
			("boosts", opponent_boosts),
			("volatiles", opponent_volatiles),
			]:
			if type(var) == list:
				state += var
				ind_dict["opponent_pokemon_active_{}".format(n)] = list(range(index, index + len(var)))
				index += len(var)
			elif type(var) in [int, float]:
				state.append(var)
				ind_dict["opponent_pokemon_active_{}".format(n)] = index
				index += 1
			else:
				print("???")
				print(n, var)
				sys.exit(1)

		opponent_side_conditions = [0] * len(relevant_conditions.side_conditions)
		opponent_active_side_conditions = battle.opponent_side_conditions
		opponent_side_conditions = [1 if x in opponent_active_side_conditions else 0 for x in relevant_conditions.side_conditions]

		state += opponent_side_conditions
		ind_dict["opponent_side_conditions"] = list(range(index, index + len(opponent_side_conditions)))
		index += len(opponent_side_conditions)

		weather = relevant_conditions.weathers.index(battle.weather)

		state.append(weather)
		ind_dict["weather"] = index
		index += 1

		fields = [0] * len(relevant_conditions.fields)
		active_fields = battle.fields
		fields = [1 if x in active_fields else 0 for x in relevant_conditions.fields ]

		fields += fields
		ind_dict["fields"] = list(range(index, index + len(fields)))
		index += len(fields)


		self.field_to_idx = ind_dict
		return state





	def compute_reward(self, battle) -> float:
		return self.reward_computing_helper(
			battle, fainted_value=25, hp_value=1, victory_value=50
		)


class StabSimpleRLPlayer(Gen8EnvSinglePlayer):
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
			len([mon for mon in battle.team.values if mon.fainted]) / 6
		)
		remaining_mon_opponent = (
			len([mon for mon in battle.opponent_team.values if mon.fainted]) / 6
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
			battle, fainted_value=0, hp_value=0, victory_value=0
		)



class SimpleRLPlayer(Gen8EnvSinglePlayer):
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
			len([mon for mon in battle.team.values if mon.fainted]) / 6
		)
		remaining_mon_opponent = (
			len([mon for mon in battle.opponent_team.values if mon.fainted]) / 6
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
