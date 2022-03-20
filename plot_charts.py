from cProfile import label
from collections import defaultdict
import os
import json
import matplotlib.pyplot as plt

measurements_dir = './measurements'
types = {'PM1': 'red', 'PM25': 'green', 'PM10': 'blue'}

measurements = defaultdict(list)

for subdir, dirs, files in os.walk(measurements_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        with open(file_path, 'r') as f:
            values = json.load(f)
            if 'history' in values:
                history = values['history']
                for value in history:
                    time = value['fromDateTime'].split('T')[1][:-5]
                    measurements[time].append(value['values'])


def mean_of(name: str, values: list) -> float:
    sum = 0
    counter = 0
    for value in values:
        v = next((v for v in value if v['name'] == name), None)
        if v is not None:
            sum += v['value']
            counter += 1
    return sum/counter


def means_of(names: list, values: list) -> dict:
    result = {}
    for name in names:
        result[name] = mean_of(name, values)
    return result


means = {k: means_of(types.keys(), v) for k, v in measurements.items()}

fig, ax = plt.subplots()
for type, colour in types.items():
    hours = map(lambda k: int(k[:2]), means.keys())
    values = map(lambda v: v[type], means.values())
    ax.scatter(list(hours), list(values), c=colour, label=type)

ax.legend()
plt.xlabel("hour")
plt.ylabel("ppm")
plt.savefig('charts/ppm_to_hour.png')
