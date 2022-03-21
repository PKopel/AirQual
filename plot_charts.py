from collections import defaultdict
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import cv2 as cv
from tqdm import tqdm

measurements_dir = './measurements/raw'
measurements_dict = defaultdict(list)
anim_dump_path = './charts/anim_dump'

measurements_df = pd.read_csv('./measurements/all_measurements.csv')
measurements_df["fromDateTime"] = measurements_df["fromDateTime"].map(pd.to_datetime)
measurements_df["tillDateTime"] = measurements_df["tillDateTime"].map(pd.to_datetime)


for subdir, dirs, files in os.walk(measurements_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        with open(file_path, 'r') as f:
            values = json.load(f)
            if 'history' in values:
                history = values['history']
                for value in history:
                    time = value['fromDateTime']
                    m = {}
                    for v in value['values']:
                        m[v['name']] = v['value']
                    measurements_dict[time].append(m)

measurements_list = []

for _, m in measurements_dict.items():
    measurements_list += m


###################################################################
# mean ppm to time of day chart
###################################################################
print("Plotting mean ppm to time of day chart")

def drop_dates(measurements: dict) -> dict:
    new_measurements = defaultdict(list)
    for k, v in measurements.items():
        time = k.split('T')[1][:-5]
        new_measurements[time] += v
    return new_measurements


def mean_of(name: str, values: list) -> float:
    sum = 0
    counter = 0
    for value in values:
        if name in value:
            sum += value[name]
            counter += 1
    return sum/counter


def means_of(names: list, values: list) -> dict:
    result = {}
    for name in names:
        result[name] = mean_of(name, values)
    return result


types = {'PM1': 'red', 'PM25': 'green', 'PM10': 'blue'}

hour_measurements = drop_dates(measurements_dict)

means = {k: means_of(types.keys(), v) for k, v in hour_measurements.items()}

fig, ax = plt.subplots()
for type, colour in tqdm(types.items()):
    hours = map(lambda k: int(k[:2]), means.keys())
    values = map(lambda v: v[type], means.values())
    ax.scatter(list(hours), list(values), c=colour, label=type)
    ax.legend()
    plt.title('Mean ppm values in 24h')
    plt.xlabel("hour")
    plt.ylabel("mean ppm")
    plt.savefig('charts/ppm_to_hour.png')
    plt.close('all')



###################################################################
# ppm to wind force chart
###################################################################
print("Plotting ppm to wind force chart")

type_values = defaultdict(list)
humidity = 'WIND_SPEED'

for v in measurements_list:
    hum = v.get(humidity)
    if hum is not None:
        for type in types.keys():
            type_value = v.get(type)
            if type_value is not None:
                type_values[type].append((hum, type_value))


for type, colour in tqdm(types.items()):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = map(lambda x: x[0], type_values[type])
    ys = map(lambda y: y[1], type_values[type])
    ax.scatter(list(xs), list(ys), s=(200./fig.dpi)**2, c=colour, label=type)
    plt.title(f'{type} ppm value to wind speed')
    plt.xlabel('wind speed')
    plt.ylabel('ppm')
    plt.savefig(f'charts/{type}_ppm_to_wind.png')
    plt.close('all')


###################################################################
# ppm to humidity chart
###################################################################
print("Plotting ppm to humidity chart")

type_values = defaultdict(list)
humidity = 'HUMIDITY'

for v in measurements_list:
    hum = v.get(humidity)
    if hum is not None:
        for type in types.keys():
            type_value = v.get(type)
            if type_value is not None:
                type_values[type].append((hum, type_value))


for type, colour in tqdm(types.items()):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = map(lambda x: x[0], type_values[type])
    ys = map(lambda y: y[1], type_values[type])
    ax.scatter(list(xs), list(ys), s=(200./fig.dpi)**2, c=colour, label=type)
    plt.title(f'{type} ppm value to humidity')
    plt.xlabel('humidity [%]')
    plt.ylabel('ppm')
    plt.savefig(f'charts/{type}_ppm_to_humidity.png')
    plt.close('all')
    
    
###################################################################
# hourly anim map
###################################################################
print("Plotting hourly anim map")

mtypes = ["PM1", "PM10", "PM25", "humidity"]
df = measurements_df[np.logical_and(pd.notna(measurements_df["longitude"]),pd.notna(measurements_df["latitude"]))]
df.loc[:,"hour"] = df["fromDateTime"].map(lambda x: x.hour)
df = df.groupby(["hour"],dropna = True)

cracow_im = cv.imread('./charts/cracow.png')

for mtype in tqdm(mtypes):
    for i in tqdm(range(24)):
        curr_h_df = df.get_group(i).groupby("sensorId").mean()
        fig, ax = plt.subplots()
        
        #hard fixed image coordinates
        plt.imshow(cracow_im,extent=[19.70, 20.20, 49.90, 50.20])
        plt.xlim([19.70, 20.20])
        plt.ylim([49.90, 50.20])
        ax.set_aspect(503.0/317.0)# rough ratio estimate from screenshot
        
        ax.scatter(curr_h_df["longitude"],curr_h_df["latitude"],s=curr_h_df[mtype])
        
        plt.xlabel("longitude")
        plt.ylabel("lattitude")
        plt.title(f"{mtype} hourly average")
        plt.text(20.06,50.17,f"{(i+2)%24}:00", {"fontsize" : 20})
        plt.savefig(f'{anim_dump_path}/{mtype}_avg_by_h_{i}.png',dpi = 400)
        plt.close('all')
        
    with imageio.get_writer(f'./charts/{mtype}_avg_by_h.gif', mode='I', fps = 4) as writer:
        for filename in [f'{anim_dump_path}/{mtype}_avg_by_h_{i}.png' for i in range(24)]:
            image = imageio.imread(filename)
            writer.append_data(image)