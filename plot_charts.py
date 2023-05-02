import os
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import imageio
import seaborn as sns
import cv2 as cv
from tqdm import tqdm

# csv columns:
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
ELEVATION = 'elevation'
SENSOR_ID = 'sensorId'
FROM_DATE_TIME = 'fromDateTime'
TILL_DATE_TIME = 'tillDateTime'
PM1 = 'PM1'
PM25 = 'PM25'
PM10 = 'PM10'
PRESSURE = 'pressure'
HUMIDITY = 'humidity'
TEMP = 'temperature'
WIND_SPEED = 'wind_speed'
WIND_BEARING = 'wind_bearing'
# additional df columns:
DATE = 'date'
HOUR = 'hour'
WEEKDAY = 'weekday'
MONTH = 'month'

TYPES = {PM1: 'red', PM25: 'green', PM10: 'blue'}

anim_dump_dir = './charts/anim_dump'

if not os.path.exists(anim_dump_dir):
    os.makedirs(anim_dump_dir)

measurements_df = pd.DataFrame()

for q_csv in glob.glob('./measurements/q*'):
    q_data = pd.read_csv(q_csv)
    measurements_df = pd.concat([measurements_df, q_data])

measurements_df[FROM_DATE_TIME] = measurements_df[FROM_DATE_TIME].map(
    pd.to_datetime)
measurements_df[TILL_DATE_TIME] = measurements_df[TILL_DATE_TIME].map(
    pd.to_datetime)
measurements_df.loc[:, HOUR] = measurements_df[FROM_DATE_TIME].map(
    lambda x: x.hour)
measurements_df.loc[:, DATE] = measurements_df[FROM_DATE_TIME].map(
    lambda x: x.date)
measurements_df.loc[:, WEEKDAY] = measurements_df[FROM_DATE_TIME].map(
    lambda x: x.weekday())
measurements_df[MONTH] = measurements_df[FROM_DATE_TIME].map(
    lambda x: x.month)
measurements_df.drop([FROM_DATE_TIME, TILL_DATE_TIME], axis=1, inplace=True)

all_charts = {}

###################################################################
# mean ppm to time of day chart
###################################################################


def ppm_to_hour():
    print('''
###################################################################
    Plotting mean ppm to time of day chart:''')

    meas_by_h = measurements_df.groupby([HOUR], dropna=True)

    fig, ax = plt.subplots()
    for type, colour in tqdm(TYPES.items()):
        hours = list(range(24))
        values = []
        for hour in tqdm(hours):
            values += [meas_by_h.get_group(hour)[type].mean()]
        ax.scatter(hours, values, c=colour, label=type)
    ax.legend()
    plt.title('Mean ppm values in 24h')
    plt.xlabel(HOUR)
    plt.ylabel("mean ppm")
    plt.savefig('charts/ppm_to_hour.png')
    plt.close('all')


all_charts['hour'] = ppm_to_hour

###################################################################
# mean ppm to time of day on a week/weekend day
###################################################################


def ppm_to_hour_day(week: bool = True):
    print('''
###################################################################
    Plotting mean ppm to time of day chart:''')

    meas_by_h = measurements_df.groupby([WEEKDAY, HOUR], dropna=True)

    days, title, file = ([0, 1, 2, 3, 4], 'a week day', 'week_day') if week\
        else ([5, 6], 'weekend', 'weekend')

    def filtr(k): return k[0] in days

    fig, ax = plt.subplots()
    for type, colour in tqdm(TYPES.items()):
        hours = list(range(24))
        keys = list(filter(filtr, meas_by_h.groups.keys()))
        values = {h: 0 for h in hours}
        for key in tqdm(keys):
            values[key[1]] += meas_by_h.get_group(key)[type].mean()
        values = list(map(lambda x: x/len(days), list(values.values())))
        ax.scatter(hours, values, c=colour, label=type)
    ax.legend()
    plt.title(f'Mean ppm values in 24h on {title}')
    plt.xlabel(HOUR)
    plt.ylabel("mean ppm")
    plt.savefig(f'charts/ppm_to_hour_{file}.png')
    plt.close('all')


all_charts['wd'] = ppm_to_hour_day
all_charts['we'] = lambda: ppm_to_hour_day(False)


###################################################################
# mean ppm to time of day over a week chart
###################################################################


def day_name(i: int) -> str:
    names = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    return names[i]


def ppm_to_hour_week(vertical: bool = True):
    print('''
###################################################################
    Plotting mean ppm to time of day over a week chart:''')

    meas_by_h = measurements_df.groupby([WEEKDAY, HOUR], dropna=True)

    figsize = (10, 30) if vertical else (30, 10)

    fig, ax = plt.subplots(figsize=figsize)
    for type, colour in tqdm(TYPES.items()):
        keys = meas_by_h.groups.keys()
        values = []
        for key in tqdm(keys):
            values += [meas_by_h.get_group(key)[type].mean()]
        nkeys = list(map(lambda k: (day_name(k[0]), k[1]), keys))
        yticks = [f'{n} {h}' for n, h in nkeys]
        vkeys = list(map(lambda k: k[1] + 24 * k[0], keys))
        if vertical:
            ax.scatter(values, vkeys, c=colour, label=type)
            plt.yticks(vkeys, yticks)
        else:
            ax.scatter(vkeys, values, c=colour, label=type)
            plt.xticks(vkeys, yticks, rotation=90)

    ax.legend()
    plt.title('Mean ppm values over a week')
    if vertical:
        plt.ylabel(HOUR)
        plt.ylim(0, 23+24*6)
        plt.gca().invert_yaxis()
        plt.xlabel("mean ppm")
        plt.savefig('charts/hour_week_to_ppm.png', bbox_inches='tight')
    else:
        plt.xlabel(HOUR)
        plt.xlim(0, 23+24*6)
        plt.ylabel("mean ppm")
        plt.savefig('charts/ppm_to_hour_week.png', bbox_inches='tight')
    plt.close('all')


all_charts['week'] = ppm_to_hour_week
all_charts['week_'] = lambda: ppm_to_hour_week(False)

###################################################################
# mean ppm to time of day by months
###################################################################


def ppm_to_hour_day_months():
    print('''
###################################################################
    Plotting mean ppm to time of day chart:''')

    meas_by_h = measurements_df.groupby([MONTH, HOUR], dropna=True)

    def filtr(m): return lambda k: k[0] == m

    fig, ax = plt.subplots()
    for month in tqdm(measurements_df[MONTH].unique()):
        hours = list(range(24))
        keys = list(filter(filtr(month), meas_by_h.groups.keys()))
        values = {h: 0 for h in hours}
        for key in tqdm(keys):
            values[key[1]] += meas_by_h.get_group(key)['PM10'].mean()
        values = list(map(lambda x: x, list(values.values())))
        ax.scatter(hours, values, color=cm.Set3(month), label=month)
    ax.legend()
    plt.title(f'Mean PM10 ppm values in 24h by months')
    plt.xlabel(HOUR)
    plt.ylabel("mean ppm")
    plt.savefig(f'charts/ppm_to_hour_months.png')
    plt.close('all')


all_charts['month'] = ppm_to_hour_day_months

###################################################################
# ppm to wind force chart
###################################################################


def ppm_to_wind_speed():
    print('''
###################################################################
    Plotting ppm to wind speed chart:''')

    for type, colour in tqdm(TYPES.items()):
        fig, ax = plt.subplots(figsize=(10, 10))
        xs = measurements_df[WIND_SPEED]
        ys = measurements_df[type]
        ax.scatter(xs, ys, s=(
            200./fig.dpi)**2, c=colour, label=type)

        plt.title(f'{type} ppm value to wind speed')
        plt.xlabel('wind speed')
        plt.ylabel('ppm')
        plt.savefig(f'charts/{type}_ppm_to_wind_speed.png')
        plt.close('all')


all_charts['ws'] = ppm_to_wind_speed

###################################################################
# wind speed histogram
###################################################################


def wind_speed_hist():
    print('''
###################################################################
    Plotting wind speed histogram:''')

    plt.figure(figsize=(10, 10))
    xs = measurements_df[WIND_SPEED]
    plt.hist(xs)

    plt.title('wind speed histogram')
    plt.xlabel('wind speed')
    plt.xlim(left=0)
    plt.savefig(f'charts/wind_speed_hist.png')
    plt.close('all')


all_charts['ws_hist'] = wind_speed_hist

###################################################################
# ppm to wind bearing chart
###################################################################


def ppm_to_wind_bearing():
    print('''
###################################################################
    Plotting ppm to wind bearing chart:''')

    for type, colour in tqdm(TYPES.items()):
        fig, ax = plt.subplots(figsize=(10, 10))
        xs = measurements_df[WIND_BEARING]
        ys = measurements_df[type]
        ax.scatter(xs, ys, s=(
            200./fig.dpi)**2, c=colour, label=type)

        plt.title(f'{type} ppm value to wind bearing')
        plt.xlabel('wind bearing')
        plt.ylabel('ppm')
        plt.savefig(f'charts/{type}_ppm_to_wind_bearing.png')
        plt.close('all')


all_charts['wb'] = ppm_to_wind_bearing

###################################################################
# wind bearing histogram
###################################################################


def wind_bearing_hist():
    print('''
###################################################################
    Plotting wind bearing histogram:''')

    plt.figure(figsize=(10, 10))
    xs = measurements_df[WIND_BEARING]
    plt.hist(xs)

    plt.title('wind bearing histogram')
    plt.xlabel('wind bearing')
    plt.xlim(0, 360)
    plt.savefig(f'charts/wind_bearing_hist.png')
    plt.close('all')


all_charts['wb_hist'] = wind_bearing_hist

###################################################################
# ppm to humidity chart
###################################################################


def ppm_to_hum():
    print('''
###################################################################
    Plotting ppm to humidity chart:''')

    for type, colour in tqdm(TYPES.items()):
        fig, ax = plt.subplots(figsize=(10, 10))
        xs = measurements_df[HUMIDITY]
        ys = measurements_df[type]
        ax.scatter(list(xs), list(ys), s=(
            200./fig.dpi)**2, c=colour, label=type)

        plt.title(f'{type} ppm value to humidity')
        plt.xlabel('humidity [%]')
        plt.ylabel('ppm')
        plt.savefig(f'charts/{type}_ppm_to_humidity.png')
        plt.close('all')


all_charts['hum'] = ppm_to_hum

###################################################################
# humidity histogram
###################################################################


def hum_hist():
    print('''
###################################################################
    Plotting humidity histogram:''')

    plt.figure(figsize=(10, 10))
    xs = measurements_df[HUMIDITY]
    plt.hist(xs)

    plt.title('humidity histogram')
    plt.xlabel('humidity [%]')
    plt.xlim(0, 100)
    plt.savefig(f'charts/humidity_hist.png')
    plt.close('all')


all_charts['hum_hist'] = hum_hist

###################################################################
# correlation heatmap
###################################################################


def corr_heatmap():
    print('''
###################################################################
    Plotting correlation heatmap:''')

    colormap = plt.cm.viridis
    data = measurements_df.iloc[:, 1:]
    data.drop([DATE], axis=1, inplace=True)
    plt.figure(figsize=(12, 12))
    sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.savefig(f'charts/corr_heatmap.png')
    plt.close('all')


all_charts['corr'] = corr_heatmap

###################################################################
# PCA biplot
###################################################################


def pca_biplot():
    print('''
###################################################################
    Plotting PCA biplot:''')

    data = measurements_df.iloc[:, 1:]
    data = data.dropna()
    data.drop([DATE], axis=1, inplace=True)
    data_n = (data-data.mean())/data.std()
    array = data_n.values
    pca = PCA()
    X_pca = pca.fit_transform(array)
    X_pca /= 10
    points = pd.DataFrame(data['month'])
    points['x'] = X_pca[:, 0]
    points['y'] = X_pca[:, 1]

    plt.figure(figsize=(18, 18))
    for m in data['month'].unique():
        m_points = points[points['month'] == m]
        plt.scatter(m_points['x'], m_points['y'],
                    color=cm.Set3(m), label=m, s=1)

    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    comps_zipped = zip(pca.components_[0], pca.components_[
                       1], data.columns.values)

    for i, comps in enumerate(comps_zipped):
        x, y, trait = comps
        length = np.sqrt(x**2 + y**2)
        x *= 3/(4*length)
        y *= 3/(4*length)
        plt.arrow(0, 0, x, y, head_width=0.01, head_length=0.02)
        x *= 9/8
        x -= 0.02
        y *= 9/8
        plt.text(x, y, trait, fontsize=14, c='blue')

    plt.savefig('charts/pca_biplot.png')
    plt.close('all')


all_charts['pca'] = pca_biplot

###################################################################
# hourly anim map
###################################################################


def h_anim_map():
    print('''
###################################################################
    Plotting hourly anim map:''')

    mtypes = [PM1, PM10, PM25, HUMIDITY]
    df = measurements_df[np.logical_and(
        pd.notna(measurements_df[LONGITUDE]), pd.notna(measurements_df[LATITUDE]))]
    df.drop([DATE], axis=1, inplace=True)
    df = df.groupby([HOUR], dropna=True)

    cracow_im = cv.imread('./charts/cracow.png')

    for mtype in tqdm(mtypes):
        for i in tqdm(range(24)):
            curr_h_df = df.get_group(i).groupby("sensorId").mean()
            fig, ax = plt.subplots()

            # hard fixed image coordinates
            plt.imshow(cracow_im, extent=[19.70, 20.20, 49.90, 50.20])
            plt.xlim([19.70, 20.20])
            plt.ylim([49.90, 50.20])
            ax.set_aspect(503.0/317.0)  # rough ratio estimate from screenshot

            ax.scatter(curr_h_df[LONGITUDE],
                       curr_h_df[LATITUDE],
                       s=curr_h_df[mtype]*2)

            plt.xlabel(LONGITUDE)
            plt.ylabel(LATITUDE)
            plt.title(f"{mtype} hourly average")
            plt.text(20.06, 50.17, f"{(i+2)%24}:00", {"fontsize": 20})
            plt.savefig(f'{anim_dump_dir}/{mtype}_avg_by_h_{i}.png', dpi=100)
            plt.close('all')

        with imageio.get_writer(f'./charts/{mtype}_avg_by_h.gif', mode='I', duration=250) as writer:
            for filename in [f'{anim_dump_dir}/{mtype}_avg_by_h_{i}.png' for i in range(24)]:
                image = imageio.v2.imread(filename)
                writer.append_data(image)


all_charts['anim'] = h_anim_map


###################################################################
# main
###################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        for chart in all_charts.values():
            chart()
    else:
        for arg in sys.argv[1:]:
            if arg in all_charts.keys():
                all_charts[arg]()
