env_player = BigBoyRLPlayer(
	player_configuration=PlayerConfiguration("SimpleRLPlayer", None),
	battle_format="gen8randombattle",
	server_configuration=LocalhostServerConfiguration,
)

opponent = RandomPlayer(
	player_configuration=PlayerConfiguration("Random player", None),
	battle_format="gen8randombattle",
	server_configuration=LocalhostServerConfiguration,
)



second_opponent = MaxDamagePlayer(
	player_configuration=PlayerConfiguration("Max damage player", None),
	battle_format="gen8randombattle",
	server_configuration=LocalhostServerConfiguration,
)

third_opponent = SimpleHeuristicsPlayer(
	player_configuration=PlayerConfiguration("Simple heuristic player", None),
	battle_format="gen8randombattle",
	server_configuration=LocalhostServerConfiguration,
)
