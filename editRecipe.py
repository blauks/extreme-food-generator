import json

data = json.loads(open("recipes.json","r").read())

for recipe in data:
    steps = recipe["steps"]
    recipe["steps"] = [step[3:].strip() for step in steps]

file = open("recipestest.json", "w")
file.write(json.dumps(data, indent=4, ensure_ascii=False))