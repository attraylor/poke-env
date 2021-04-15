from poke_env.environment.status import Status
from poke_env.environment.weather import Weather
from poke_env.environment.field import Field
from poke_env.environment.effect import Effect
from poke_env.environment.side_condition import SideCondition


statuses = [None, Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]
weathers = [None, Weather.DESOLATELAND,Weather.DELTASTREAM,Weather.HAIL,Weather.PRIMORDIALSEA,Weather.RAINDANCE,Weather.SANDSTORM,Weather.SUNNYDAY]

volatiles = [
	Effect.ATTRACT,
	Effect.CONFUSION,
	Effect.CURSE,
	Effect.DESTINY_BOND,
	Effect.DISABLE, #Todo: counter
	Effect.ELECTRIC_TERRAIN, #TODO: IRRELEVANT??!?!?!
	Effect.ENCORE,
	Effect.LEECH_SEED,
	Effect.MISTY_TERRAIN, #TODO: IRRELEVANT??!?!?!
	Effect.PHANTOM_FORCE,
	Effect.PSYCHIC_TERRAIN, #TODO: IRRELEVANT??!?!?!
	Effect.SAFEGUARD, #TODO: IRRELEVANT??!?!?!
	Effect.STICKY_WEB,
	Effect.SUBSTITUTE,
	Effect.TAUNT,
	Effect.TORMENT,
	Effect.YAWN,
]

side_conditions = [
	SideCondition.AURORA_VEIL,
	SideCondition.LIGHT_SCREEN,
	SideCondition.LUCKY_CHANT,
	SideCondition.MIST,
	SideCondition.REFLECT,
	SideCondition.SAFEGUARD,
	SideCondition.SPIKES,
	SideCondition.STEALTH_ROCK,
	SideCondition.STICKY_WEB,
	SideCondition.TAILWIND,
	SideCondition.TOXIC_SPIKES,
	SideCondition.G_MAX_CANNONADE ,
    SideCondition.G_MAX_STEELSURGE ,
    SideCondition.G_MAX_VINE_LASH ,
    SideCondition.G_MAX_VOLCALITH ,
    SideCondition.G_MAX_WILDFIRE ,
]

fields = [
	Field.ELECTRIC_TERRAIN,
	Field.GRASSY_TERRAIN,
	Field.GRAVITY,
	Field.MISTY_TERRAIN,
	Field.PSYCHIC_TERRAIN,
	Field.TRICK_ROOM,
]
