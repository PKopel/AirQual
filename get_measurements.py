import json
import requests
import datetime
import pathlib
import random

with open('sensors_list.json', 'r') as f:
    sensors = json.load(f)

with open('Airly_API.json', 'r') as f:
    api_keys = json.load(f)
    n_api_keys = len(api_keys)

date = datetime.datetime.now()
dir = f'measurements/{date.day}-{date.month}-{date.year}'

pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

ids = list(map(lambda s: s['id'], sensors))
random.shuffle(ids)

url = 'https://airapi.airly.eu/v2/measurements/installation'
params = {'indexType': 'AIRLY_CAQI',
          'installationId': '',
          'includeWind': 'true'}
headers = {'Accept': 'application/json',
           'Accept-Language': 'en',
           'apikey': ''}


for i, id in enumerate(ids):
    api_key = api_keys[i % n_api_keys]

    params['installationId'] = id
    headers['apikey'] = api_key

    result = requests.get(url, params, headers=headers)

    if len(result.json()) > 1:
        with open(f'{dir}/{id}.json', 'w') as f:
            json.dump(result.json(), f, indent=4, sort_keys=True)

        print(id)
    else:
        break
