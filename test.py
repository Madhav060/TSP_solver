import sys, os
sys.path.append(os.path.dirname(__file__))

from data_generator import load_tsp_file

cities = load_tsp_file("tsp_data/gr48.tsp")
print(len(cities))
for c in cities[:5]:
    print(c.x, c.y)
