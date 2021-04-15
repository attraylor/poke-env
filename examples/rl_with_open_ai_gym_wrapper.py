# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
<<<<<<< HEAD
=======
from poke_env.data import MOVES

print(tf.__version__)
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.player.player import Player
>>>>>>> wip

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents import DQNAgent as DQN_default
from rl.policy import EpsGreedyQPolicy as EPSGREEDY_default
from poke_env.dqn2 import DQNAgent
from rl.policy import LinearAnnealedPolicy
from poke_env.policy2 import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import pickle

# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
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


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 10000

tf.random.set_seed(0)
np.random.seed(0)


# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps):
	dqn.fit(player, nb_steps=nb_steps)
	player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes):
	# Reset battle statistics
	player.reset_battles()
	dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

	print(
		"DQN Evaluation: %d victories out of %d episodes"
		% (player.n_won_battles, nb_episodes)
	)

def build_dqn(input_shape=10, name=""):
	# Output dimension
	n_action = len(env_player.action_space)

	model = Sequential()
	model.add(Dense(128, activation="elu", input_shape=(1, input_shape)))

	# Our embedding have shape (1, 10), which affects our hidden layer
	# dimension and output dimension
	# Flattening resolve potential issues that would arise otherwise
	#model.add(Flatten())
	model.add(Dense(64, activation="elu"))
	model.add(Dense(n_action, activation="linear"))

	memory = SequentialMemory(limit=10000, window_length=1)

	if name == "old":
		# Simple epsilon greedy
		policy = LinearAnnealedPolicy(
			EPSGREEDY_default(),
			attr="eps",
			value_max=1.0,
			value_min=0.05,
			value_test=0,
			nb_steps=10000,
		)

		# Defining our DQN
		dqn = DQN_default(
			model=model,
			nb_actions=18,
			policy=policy,
			memory=memory,
			nb_steps_warmup=1000,
			gamma=0.5,
			target_model_update=1,
			delta_clip=0.01,
			enable_double_dqn=True,
		)
	else:
		# Simple epsilon greedy
		policy = LinearAnnealedPolicy(
			EpsGreedyQPolicy(),
			attr="eps",
			value_max=1.0,
			value_min=0.05,
			value_test=0,
			nb_steps=10000,
		)

		# Defining our DQN
		dqn = DQNAgent(
			model=model,
			nb_actions=18,
			policy=policy,
			memory=memory,
			nb_steps_warmup=1000,
			gamma=0.5,
			target_model_update=1,
			delta_clip=0.01,
			enable_double_dqn=True,
		)

	dqn.compile(Adam(lr=0.00025), metrics=["mae"])
	return dqn


def build_advanced_dqn(input_shape=4, name=""):
	# Output dimension
	move_emb_dim = 32
	n_action = len(env_player.action_space)

	model = Sequential()
	model.add(Embedding(796, move_emb_dim, 4)) #MOVE EMBEDDING
	model.add(Flatten())
	model.add(Dense(128, activation="elu", input_shape=(1, move_emb_dim * 4)))

	# Our embedding have shape (1, 10), which affects our hidden layer
	# dimension and output dimension
	# Flattening resolve potential issues that would arise otherwise
	model.add(Flatten())
	model.add(Dense(64, activation="elu"))
	model.add(Dense(n_action, activation="linear"))

	memory = SequentialMemory(limit=10000, window_length=1)

	if name == "old":
		# Simple epsilon greedy
		policy = LinearAnnealedPolicy(
			EPSGREEDY_default(),
			attr="eps",
			value_max=1.0,
			value_min=0.05,
			value_test=0,
			nb_steps=10000,
		)

		# Defining our DQN
		dqn = DQN_default(
			model=model,
			nb_actions=18,
			policy=policy,
			memory=memory,
			nb_steps_warmup=1000,
			gamma=0.5,
			target_model_update=1,
			delta_clip=0.01,
			enable_double_dqn=True,
		)
	else:
		# Simple epsilon greedy
		policy = LinearAnnealedPolicy(
			EpsGreedyQPolicy(),
			attr="eps",
			value_max=1.0,
			value_min=0.05,
			value_test=0,
			nb_steps=10000,
		)

		# Defining our DQN
		dqn = DQNAgent(
			model=model,
			nb_actions=18,
			policy=policy,
			memory=memory,
			nb_steps_warmup=1000,
			gamma=0.5,
			target_model_update=1,
			delta_clip=0.01,
			enable_double_dqn=True,
		)

	dqn.compile(Adam(lr=0.00025), metrics=["mae"])
	return dqn


if __name__ == "__main__":
    env_player = SimpleRLPlayer(battle_format="gen8randombattle")

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    # Output dimension
    n_action = len(env_player.action_space)

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=10000, window_length=1)

    # Ssimple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
