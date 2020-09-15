import sys

#usage: python clean_js_to_json.py js_path json_path
with open(sys.argv[2], "w+") as wf:
	wf.write("{\n")
	with open(sys.argv[1]) as rf:
		for line in rf:
			l = line.rstrip()
			if line[0:2] == "	\"":
				wf.write(l)
				wf.write("\n")
			elif line[0:7] == "		name:":
				wf.write("		\"name\":" + l[7:])
				wf.write("\n")
			elif line[0:7] == "		desc:":
				wf.write("		\"desc\":" + l[7:-1])
				wf.write("\n")
			elif line[0:6] == "		num:":
				wf.write("		\"num\":" + l[6:])
				wf.write("\n")
			elif line[0:7] == "		id:":
				wf.write("		\"id\":" + l[5:])
				wf.write("\n")
			elif line[0:3] == "	},":
				wf.write(l)
				wf.write("\n")
	wf.write("}\n")
