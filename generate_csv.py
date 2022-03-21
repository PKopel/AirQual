from collections import defaultdict
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from numba import jit
from tqdm import tqdm

    
def make_measurements_csv(measurements_dir):
    columns = ["sensorId","latitude","longitude","elevation","fromDateTime","tillDateTime","PM1","PM25","PM10","pressure","humidity","temperature","wind_speed","wind_bearing"]
    data = []
    counter = 0
    
    sensor_data = {}

    with open('./sensors_list.json', 'r') as f:
        sensor_json = json.load(f)
        for sensor in sensor_json:
            sensor_data[sensor["id"]] = sensor    
    
    print("generating csv")
    for subdir, dirs, files in tqdm(list(os.walk(measurements_dir))):
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r') as f:
                values = json.load(f)
                if 'history' in values:
                    history = values['history']
                    for value in history:
                        

                        new_entry = [None for _ in columns]

                        sensor_id = int(file.split('.')[0])
                        new_entry[columns.index("sensorId")] = sensor_id
                        new_entry[columns.index("fromDateTime")] = pd.to_datetime(value["fromDateTime"])
                        new_entry[columns.index("tillDateTime")] = pd.to_datetime(value["tillDateTime"])
                        if(sensor_id in sensor_data):
                            new_entry[columns.index("latitude")] = sensor_data[sensor_id]["location"]["latitude"]
                            new_entry[columns.index("longitude")] = sensor_data[sensor_id]["location"]["longitude"]
                            new_entry[columns.index("elevation")] = sensor_data[sensor_id]["elevation"]
                        
                        m = {}
                        for v in value['values']:
                            m[v['name']] = v['value']
                            
                        if("PM1" in m):
                            new_entry[columns.index("PM1")] = m["PM1"]
                        if("PM10" in m):
                            new_entry[columns.index("PM10")] = m["PM10"]
                        if("PM25" in m):
                            new_entry[columns.index("PM25")] = m["PM25"]
                        if("PRESSURE" in m):
                            new_entry[columns.index("pressure")] = m["PRESSURE"]
                        if("HUMIDITY" in m):
                            new_entry[columns.index("humidity")] = m["HUMIDITY"]
                        if("TEMPERATURE" in m):
                            new_entry[columns.index("temperature")] = m["TEMPERATURE"]
                        if("WIND_SPEED" in m):
                            new_entry[columns.index("wind_speed")] = m["WIND_SPEED"]
                        if("WIND_BEARING" in m):
                            new_entry[columns.index("wind_bearing")] = m["WIND_BEARING"]

                        data.append(new_entry)
    return pd.DataFrame(data = data, columns = columns)

if(__name__ == "__main__"):
    data = make_measurements_csv('./measurements/raw')
    print(data)
    save_path = "./measurements/all_measurements.csv"
    data.to_csv(save_path)
    print(f"csv file saved to {save_path}")
    

                    
                    
