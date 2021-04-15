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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
from models import *
from players import *


from config_utils import create_config

from poke_env.teambuilder.teambuilder import Teambuilder

class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.teams = [self.join_team(self.parse_showdown_team(team)) for team in teams]

    def yield_team(self):
        return np.random.choice(self.teams)


if __name__ == "__main__":
    global config
    hyperparameter_defaults = dict(
        experiment_name = "BigBoy",
        batch_size = 200,
        batch_cap = 2,
        optimize_every = 1000,
        gamma = .5,
        eps_start = .9,
        eps_end = .05,
        eps_decay = 200,
        target_update = 5,
        learning_rate = 0.00025,
        memory_size = 10000,
        nb_training_steps = 5000,
        nb_evaluation_episodes = 100,
        species_emb_dim = 3,
        move_emb_dim = 3,
        item_emb_dim = 1,
        ability_emb_dim = 1,
        type_emb_dim = 3,
        status_emb_dim = 1,
        weather_emb_dim = 1,
        pokemon_embedding_hidden_dim = 4,
        team_embedding_hidden_dim = 4,
        move_encoder_hidden_dim = 3,
        opponent_hidden_dim = 3,
        complete_state_hidden_dim = 5,
        complete_state_output_dim = 22,
        seed = 420,
        n_layers = 5
    )

    team_1 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish

Cinderace (M) @ Life Orb
Ability: Blaze
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Pyro Ball
- Sucker Punch
- U-turn
- High Jump Kick

Toxtricity (M) @ Throat Spray
Ability: Punk Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Overdrive
- Boomburst
- Shift Gear
- Fire Punch

Seismitoad (M) @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Stealth Rock
- Scald
- Earthquake
- Toxic

Corviknight (M) @ Leftovers
Ability: Pressure
EVs: 248 HP / 80 SpD / 180 Spe
Impish Nature
- Defog
- Brave Bird
- Roost
- U-turn
"""

    custom_builder = RandomTeamFromPool([team_1])

    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    writepath = os.path.join("results/",config.experiment_name)
    if not os.path.exists(writepath):
        os.makedirs(writepath)



    env_player = BigBoyRLPlayer(
        player_configuration=PlayerConfiguration("SimpleRLPlayer", None),
        battle_format="gen8ou",
        server_configuration=LocalhostServerConfiguration,
        team = custom_builder
    )

    opponent = RandomPlayer(
        player_configuration=PlayerConfiguration("Random player", None),
        battle_format="gen8ou",
        server_configuration=LocalhostServerConfiguration,
        team = custom_builder
    )



    second_opponent = MaxDamagePlayer(
        player_configuration=PlayerConfiguration("Max damage player", None),
        battle_format="gen8ou",
        server_configuration=LocalhostServerConfiguration,
        team = custom_builder
    )

    third_opponent = SimpleHeuristicsPlayer(
        player_configuration=PlayerConfiguration("Simple heuristic player", None),
        battle_format="gen8ou",
        server_configuration=LocalhostServerConfiguration,
        team = custom_builder
    )

    n_actions = len(env_player.action_space)


    policy_net = BigBoy_DQN(config)

    target_net = BigBoy_DQN(config)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    #optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    memory = ReplayMemory(config.memory_size)

    steps_done = 0

    loss_hist = []
    reward_hist = []

    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"nb_steps": config.nb_training_steps},
    )
    model_path = os.path.join(writepath, "saved_model.torch")
    torch.save(policy_net.state_dict(), model_path)


    for chart_name, arr in [("loss", loss_hist), ("reward", reward_hist)]:
        x = range(len(arr))
        fig, ax = plt.subplots()
        ax.plot(x, arr)
        ax.set(xlabel = "batches", ylabel = chart_name, title = "{} hist over time".format(chart_name))
        ax.grid()
        fig.savefig(os.path.join(writepath, "{}.png".format(chart_name)))
        plt.gcf().clear()


    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"nb_episodes": config.nb_evaluation_episodes},
    )

    print("Results against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"nb_episodes": config.nb_evaluation_episodes},
    )

    print("Results against simple heuristic player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=third_opponent,
        env_algorithm_kwargs={"nb_episodes": config.nb_evaluation_episodes},
    )

    print('Complete')
    #env.render()
    env_player.close()
