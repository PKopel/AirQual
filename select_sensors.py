import json
import datetime

date = datetime.datetime.now()
dir = f'measurements/{date.day}-{date.month}-{date.year}'


with open('sensors_list.json', 'r') as f:
    sensors = json.load(f)


def check_sensor(sensor):
    id = sensor['id']
    try:
        with open(f'{dir}/{id}.json', 'r') as f:
            data = json.load(f)
        if len(data) > 1:
            values = data['current']['values']
            values = map(lambda v: v['name'] == 'WIND_SPEED', values)
            return any(values)
    except:
        return True


sensors = list(filter(check_sensor, sensors))


with open(f'sensors_list.json', 'w') as f:
    json.dump(sensors, f, indent=4, sort_keys=True)

print(len(sensors))
