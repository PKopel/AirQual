import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import date

SENSORS_FILE = './data/sensors_list.json'
MEASUREMENTS_DIR = './data'
COLUMNS = ["sensorId", "latitude", "longitude", "elevation", "fromDateTime", "tillDateTime",
           "PM1", "PM25", "PM10", "pressure", "humidity", "temperature", "wind_speed", "wind_bearing"]


def make_measurements_csv(measurements_dir):
    data = []

    sensor_data = {}

    with open(SENSORS_FILE, 'r') as f:
        sensor_json = json.load(f)
        for sensor in sensor_json:
            sensor_data[sensor["id"]] = sensor

    print("generating csv")
    for subdir, _, files in tqdm(list(os.walk(measurements_dir))):
        for file in tqdm(files):
            if 'tar.gz' not in file:
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    values = json.load(f)
                    if 'history' in values:
                        history = values['history']
                        for value in history:

                            new_entry = [None for _ in COLUMNS]

                            sensor_id = int(file.split('.')[0])
                            new_entry[COLUMNS.index("sensorId")] = sensor_id
                            new_entry[COLUMNS.index("fromDateTime")] = pd.to_datetime(
                                value["fromDateTime"])
                            new_entry[COLUMNS.index("tillDateTime")] = pd.to_datetime(
                                value["tillDateTime"])
                            if (sensor_id in sensor_data):
                                new_entry[COLUMNS.index(
                                    "latitude")] = sensor_data[sensor_id]["location"]["latitude"]
                                new_entry[COLUMNS.index(
                                    "longitude")] = sensor_data[sensor_id]["location"]["longitude"]
                                new_entry[COLUMNS.index(
                                    "elevation")] = sensor_data[sensor_id]["elevation"]

                            m = {}
                            for v in value['values']:
                                m[v['name']] = v['value']

                            if ("PM1" in m):
                                new_entry[COLUMNS.index("PM1")] = m["PM1"]
                            if ("PM10" in m):
                                new_entry[COLUMNS.index("PM10")] = m["PM10"]
                            if ("PM25" in m):
                                new_entry[COLUMNS.index("PM25")] = m["PM25"]
                            if ("PRESSURE" in m):
                                new_entry[COLUMNS.index(
                                    "pressure")] = m["PRESSURE"]
                            if ("HUMIDITY" in m):
                                new_entry[COLUMNS.index(
                                    "humidity")] = m["HUMIDITY"]
                            if ("TEMPERATURE" in m):
                                new_entry[COLUMNS.index(
                                    "temperature")] = m["TEMPERATURE"]
                            if ("WIND_SPEED" in m):
                                new_entry[COLUMNS.index(
                                    "wind_speed")] = m["WIND_SPEED"]
                            if ("WIND_BEARING" in m):
                                new_entry[COLUMNS.index(
                                    "wind_bearing")] = m["WIND_BEARING"]

                            data.append(new_entry)
    return pd.DataFrame(data=data, columns=COLUMNS)


if (__name__ == "__main__"):
    quarter = (date.today().month-1)//3 + 1
    year = date.today().year
    MEASUREMENTS_CSV = f'./data/csv/q{quarter}_{year}_measurements.csv'
    new_data = make_measurements_csv(f'{MEASUREMENTS_DIR}/raw')

    if os.path.exists(MEASUREMENTS_CSV):
        old_data = pd.read_csv(MEASUREMENTS_CSV)
        old_data = old_data[COLUMNS]
        data = pd.concat([old_data, new_data])
    else:
        data = new_data

    data.to_csv(MEASUREMENTS_CSV)
    print(f"csv file saved to {MEASUREMENTS_CSV}")
