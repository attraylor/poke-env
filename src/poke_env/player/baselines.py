# -*- coding: utf-8 -*-
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer  # noqa: F401

from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import LocalhostServerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.utils import to_id_str

import numpy as np

class MaxBasePowerPlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)

class SimpleHeuristicsPlayer(Player):
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    def _estimate_matchup(self, mon, opponent):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_dynamax(self, battle, n_remaining_mons):
        if battle.can_dynamax:
            # Last full HP mon
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # Matchup advantage and full hp on full hp
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False

    def _should_switch_out(self, battle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if battle.turn == 1:
            self.switches_in_a_row = 0
            self.forced_attack = False
        elif self.switches_in_a_row > 6:
            self.switches_in_a_row = 0
            self.forced_attack = True
            return False
        else:
            self.forced_attack = False
            self.switches_in_a_row += 1
        # If there is a decent switch in...
        if [
            m
            for m in battle.available_switches
            if self._estimate_matchup(m, opponent) > 0
        ]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                self._estimate_matchup(active, opponent)
                < self.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        self.switches_in_a_row = 0
        self.forced_attack = False
        return False

    def _stat_estimation(self, mon, stat):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    def choose_move(self, battle):
        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
            opponent, "def"
        )
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
            opponent, "spd"
        )

        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )
            n_opp_remaining_mons = 6 - len(
                [m for m in battle.team.values() if m.fainted is True]
            )

            # Entry hazard...
            for move in battle.available_moves:
                # ...setup
                if (
                    n_opp_remaining_mons >= 3
                    and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    return self.create_order(move)

                # ...removal
                elif (
                    battle.side_conditions
                    and move.id in self.ANTI_HAZARDS_MOVES
                    and n_remaining_mons >= 2
                ):
                    return self.create_order(move)

            # Setup moves
            if (
                active.current_hp_fraction == 1
                and self._estimate_matchup(active, opponent) > 0
            ):
                for move in battle.available_moves:
                    if (
                        move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            [active.boosts[s] for s, v in move.boosts.items() if v > 0]
                        )
                        < 6
                    ):
                        return self.create_order(move)

            move = max(
                battle.available_moves,
                key=lambda m: m.base_power
                * (1.5 if m.type in active.types else 1)
                * (
                    physical_ratio
                    if m.category == MoveCategory.PHYSICAL
                    else special_ratio
                )
                * m.accuracy
                * m.expected_hits
                * opponent.damage_multiplier(m),
            )
            return self.create_order(
                move, dynamax=self._should_dynamax(battle, n_remaining_mons)
            )

        if battle.available_switches:
            return self.create_order(
                max(
                    battle.available_switches,
                    key=lambda s: self._estimate_matchup(s, opponent),
                )
            )

        return self.choose_random_move(battle)

class EpsilonRandomSimpleHeuristicsPlayer(SimpleHeuristicsPlayer):
    def __init__(self,
    player_configuration: Optional[PlayerConfiguration] = None,
    *,
    avatar: Optional[int] = None,
    battle_format: str = "gen8randombattle",
    log_level: Optional[int] = None,
    max_concurrent_battles: int = 1,
    server_configuration: Optional[ServerConfiguration] = None,
    start_timer_on_battle_start: bool = False,
    start_listening: bool = True,
    team: Optional[Union[str, Teambuilder]] = None,
    epsilon: int):
        super().__init__(
            player_configuration = player_configuration,
            avatar = avatar,
            battle_format = battle_format,
            log_level = log_level,
            max_concurrent_battles = max_concurrent_battles,
            server_configuration = server_configuration,
            start_timer_on_battle_start = start_timer_on_battle_start,
            start_listening = start_listening,
            team = team,
            )
        self.epsilon = epsilon

    def choose_move(self, battle):
        random_roll = np.random.rand()
        if random_roll > self.epsilon or (hasattr(self, "forced_attack") and self.forced_attack == True):
            move = super().choose_move(battle)
        else:
            move = self.choose_random_move(battle)
        return move
