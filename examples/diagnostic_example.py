# -*- coding: utf-8 -*-
import asyncio
import numpy as np

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from ou_max_player import MaxDamagePlayer


async def main():

	diagnostics = []
	team_1 = """
Lucario (M) @ Assault Vest
Ability: Justified
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
- Focus Blast
- Aura Sphere
"""
	team_2 = """
Bisharp (M) @ Life Orb
Ability: Defiant
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Brick Break"""
	diagnostics.append([team_1, team_2])
	team_1 = """
Lucario (M) @ Choice Specs
Ability: Justified
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
- Aura Sphere
"""
	team_2 = """
Lapras (M) @ Leftovers
Level: 100
Ability: Shell Armor
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
- Water Gun"""
	diagnostics.append([team_1, team_2])
	# We create two players.
	for team_1, team_2 in diagnostics:
		random_player = RandomPlayer(
			battle_format="gen8ou", team=team_2, max_concurrent_battles=10
		)
		max_damage_player = MaxDamagePlayer(
			battle_format="gen8ou", team=team_1, max_concurrent_battles=10
		)

		# Now, let's evaluate our player
		cross_evaluation = await cross_evaluate(
			[random_player, max_damage_player], n_challenges=50
		)

		print(
			"Max damage player won %d / 100 battles"
			% (cross_evaluation[max_damage_player.username][random_player.username] * 100)
		)


if __name__ == "__main__":
	asyncio.get_event_loop().run_until_complete(main())
