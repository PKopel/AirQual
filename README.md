# Airly scripts
Scripts downloading air quality data from [Airly](https://developer.airly.org/en).

## Usage
1. Create file `Airlty_API.txt` containing your Airly API key
2. Save list of sensors to query in `sensors_list.json` (as a JSON list of objects with at least `'id'` field)
3. Run `python get_measurements.py` to download and save current measurmenets