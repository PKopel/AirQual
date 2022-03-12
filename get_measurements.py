import json
import requests
import datetime
import pathlib

with open('sensors_list.json', 'r') as f:
    sensors = json.load(f)

with open('Airly_API.txt', 'r') as f:
    api_key = f.read()

date = datetime.datetime.now()
dir = f'measurements/{date.day}-{date.month}-{date.year}'

pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

ids = map(lambda s: s['id'], sensors)

url = 'https://airapi.airly.eu/v2/measurements/installation'
params = {'indexType': 'AIRLY_CAQI',
          'installationId': '',
          'includeWind': 'true'}
headers = {'Accept': 'application/json',
           'Accept-Language': 'en',
           'apikey': api_key}


for id in ids:
    params['installationId'] = id

    result = requests.get(url, params, headers=headers)

    if len(result.json()) > 1:
        with open(f'{dir}/{id}.json', 'w') as f:
            json.dump(result.json(), f, indent=4, sort_keys=True)

        print(id)
