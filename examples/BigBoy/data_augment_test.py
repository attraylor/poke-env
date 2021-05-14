import numpy as np
import copy

def data_augment(state, action, next_state, field_to_idx, k=25):
	augmented_data = []
	for i in range(k):
		new_state = copy.deepcopy(state)
		new_next_state = copy.deepcopy(next_state)
		new_party_order = np.random.permutation(5) + 2 #shuffles [2,3,4,5,6]
		for old_pokemon_idx, new_pokemon_party_idx in enumerate(new_party_order):
			old_pokemon_party_idx = old_pokemon_idx + 2
			for key, value in field_to_idx.items():
				if "pokemon_{}".format(old_pokemon_party_idx) in key:
					new_key = key.replace(str(old_pokemon_party_idx), str(new_pokemon_party_idx))
					new_key_idx = field_to_idx[new_key]
					old_key_idx = field_to_idx[key]
					if type(value) == list:
						new_state[new_key_idx[0]:new_key_idx[-1]+1] = state[old_key_idx[0]:old_key_idx[-1]+1]
						new_next_state[new_key_idx[0]:new_key_idx[-1]+1] = next_state[old_key_idx[0]:old_key_idx[-1]+1]
					else:
						new_state[new_key_idx] = state[old_key_idx]
						new_next_state[new_key_idx] = next_state[old_key_idx]
		if action >= 16:
			new_action = new_party_order[action - 16] + 16 - 2
			print(action, new_action, new_party_order[1])
		augmented_data.append((new_state, new_action, new_next_state))
	return augmented_data

#field_to_idx = {"hp": 0, "pokemon_1_x": 1, "pokemon_2_x": 2, "pokemon_3_x": 3, "pokemon_4_x": 4, "pokemon_5_x": 5,"pokemon_6_x": 6,  }
field_to_idx = {"hp": 0, "pokemon_1_x": [1,2], "pokemon_2_x": [3,4], "pokemon_3_x": [5,6], "pokemon_4_x": [7,8], "pokemon_5_x": [9,10],"pokemon_6_x": [11,12] }

state = [0.33] + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
next_state = [0.33] + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
action = 17

z = data_augment(state, action, next_state, field_to_idx)
for a, b, c in z:
	print(a, b, c)
