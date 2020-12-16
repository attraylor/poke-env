import os

def isFloat(s):
	return s.replace('.','',1).isdigit()

def isInteger(s):
	return s.isdigit()

def create_config(path):
	config_specification = {}
	with open(os.path.join(path)) as rf:
		for line in rf:
			field, value = line.strip().split("\t")
			if value == "True":
				config_specification[field] = True
			elif value == "False":
				config_specification[field] = False
			elif isInteger(value):
				config_specification[field] = int(value)
			elif isFloat(value) or ("1e" in value and "<" not in value):
				config_specification[field] = float(value)
			else:
				config_specification[field] = value

	if "seed" not in config_specification.keys():
		print("No seed set. Results not reproducible. Set seed in config")
		sys.exit(1)
	return config_specification
