from os import listdir
from pathlib import Path
from os.path import basename
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


@dataclass
class Line:
    tidal_orthogonal_projection: np.ndarray
    times: np.ndarray
    duration: float
    index: int
    name: str
    filtered_out: bool
    start_date: str
    line_dir: float
    line_course: int
    coordinates: list[np.ndarray, np.ndarray]


@dataclass
class Data:
    lines: list[Line]
    tidal_prediction: np.ndarray  # values
    tidal_time: np.ndarray  # their times


@lru_cache
def data_loading(all_p190, old_tides, future_tides, line_dir, year, thrsh):
    """
    Parameters:
    - all_p190 (str): Directory path containing all old the p190 files.
    - old_tides (str): File path for the old tides data.
    - future_tides (str): File path for the future tides data.
    - line_dir (str): Direction of old survey's line (radians).
    - thrsh (float): Minimal survey line duration.

    Returns:
    - Data object
    """
    df = pd.read_excel(old_tides)
    df = check_data(df, year)
    path_p190s = [Path(all_p190) / name for name in listdir(all_p190) if name.endswith('p190')]
    lines = [make_line(path_p190, df=df, direction=line_dir, thrsh=thrsh) for path_p190 in path_p190s]

    future_df = pd.read_excel(future_tides)
    tidal_time, tidal_prediction = prepare_future_currents(future_df, line_dir)
    tidal_time = np.array(tidal_time)
    tidal_time = tidal_time / 3600  # seconds to hours
    tidal_prediction = np.array(tidal_prediction)

    return Data(
        lines=lines,
        tidal_time=tidal_time,
        tidal_prediction=tidal_prediction
    )


def make_line(path_p190, df, direction, thrsh):
    '''
    path_p190: path to used S file
    df: currents data
    direction: direction of line in grad
    thrsh: minimal duration in hours
    '''
    # df = pd.read_excel(path_df)
    year = int(str(df['Date&Time (GMT+11h)'].values[0])[:4])
    hours, line_start, duration, easting, northing, name = p190read(path_p190, year)
    tidal_orthogonal_projection, times = make_ort_proj(df=df, hours=hours, line_dir=direction, threshold=thrsh)
    course = int(np.sign(northing[-1] - northing[0]))
    return Line(tidal_orthogonal_projection=tidal_orthogonal_projection,
                times=times,
                duration=duration,
                index=int(basename(path_p190)[14:16]),
                name=name,
                filtered_out=duration <= thrsh,
                start_date=line_start,
                line_dir=direction,
                line_course=course,
                coordinates=[easting, northing]
                )


def p190read(path_p190, year):
    easting = []
    northing = []
    hours = []
    with open(path_p190, 'r') as f:
        for line in f:
            if line[0] == 'S':
                easting.append(float(line[47:55]))
                northing.append(float(line[56:65]))
                hour = (int(line[70:73]) - 1) * 24 + int(line[73:75]) + int(line[75:77]) / 60 + int(line[77:79]) / 3600
                hours.append(hour)
            elif line[:9] == 'H2600LINE':
                name = line[15:26]

        # hours = np.array([
        #     (int(line[70:73]) - 1) * 24 + int(line[73:75]) + int(line[75:77]) / 60 + int(line[77:79]) / 3600
        #     for line in f if line.startswith('S')])
    hours = np.array(hours)
    easting = np.array(easting)
    northing = np.array(northing)
    start, duration = hours[0], hours[-1] - hours[0]
    return hours, compute_date_from_hours(start, year), duration, easting, northing, name


def check_data(df, year):
    if 'JUL_SECOND' not in df.columns:  # PA_ files
        df['Date&Time (GMT+11h)'] = pd.to_datetime(df['Date&Time (GMT+11h)'])
        df['JUL_SECOND'] = (df['Date&Time (GMT+11h)'] - pd.to_datetime(f'{year}-01-01')).dt.total_seconds() - 11 * 3600
    return df


def make_ort_proj(df, hours, line_dir, threshold=9):
    total_sec = hours * 3600
    mask = (df['JUL_SECOND'] >= total_sec[0]) & (df['JUL_SECOND'] <= total_sec[-1])
    # print(len(df['JUL_SECOND'][mask]))
    first_ind, last_ind = df['JUL_SECOND'][mask].keys()[[0, -1]]
    first_ind, last_ind = first_ind - 1, last_ind + 2
    cur_time = df[first_ind:last_ind]['JUL_SECOND'].to_numpy() / 3600
    # print(len(cur_time))

    if 'XL' in df.columns:  # ADCP
        xl = df[first_ind:last_ind]['XL']
        # xl = np.interp(total_sec, cur_time, xl)

    else:  # PA_ files
        speed = df[first_ind:last_ind]['Curr. spd (m/s)'].to_numpy()
        # speed = np.interp(total_sec, cur_time, speed)

        angle = df[first_ind:last_ind]['Curr. dir (deg towards)'].to_numpy() / 180 * np.pi
        # angle = np.interp(total_sec, cur_time, angle)

        # Find the index of the item closest to 2π
        ind1 = np.argmin(np.abs(angle - 2 * np.pi))
        if abs(angle[ind1] - 2 * np.pi) < threshold / 180 * np.pi:
            # Find the index of the item closest to 0
            ind2 = np.argmin(np.abs(angle))
            # Check the order of ind1 and ind2
            if ind1 > ind2:
                start, end = ind2, ind1  # Swap the indices
            else:
                start, end = ind1, ind2
            # Determine the middle index
            middle_ind = start + (end - start) // 2
            if ind1 < ind2:
                # Linearize the first half from angle[ind1] to 2π
                angle[start:middle_ind + 1] = np.linspace(angle[ind1], 2 * np.pi, middle_ind - start + 1)
                # Linearize the second half from 0 to angle[ind2]
                angle[middle_ind + 1:end + 1] = np.linspace(0, angle[ind2], end - middle_ind)
            else:
                #  Linearize the first half from 0 to angle[ind2]
                angle[start:middle_ind + 1] = np.linspace(angle[ind2], 0, middle_ind - start + 1)
                # Linearize the second half from 0 to angle[ind2]
                angle[middle_ind + 1:end + 1] = np.linspace(2 * np.pi, angle[ind1], end - middle_ind)
        xl = (speed * np.cos(angle - line_dir * np.pi / 180)).tolist()
    xl = np.array(xl)
    return xl, cur_time


def seconds_to_datetime(total_seconds_array, year):
    start_date = datetime(year, 1, 1)
    datetime_array = [start_date + timedelta(seconds=s) for s in total_seconds_array]
    formatted_dates = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetime_array]
    return formatted_dates


def compute_date_from_hours(total_hours, year):
    base_date = datetime(year, 1, 1)  # This sets the "start" to the date of the given year
    delta = timedelta(hours=total_hours)
    result_date = base_date + delta
    return result_date.strftime('%Y-%m-%d %H:%M:%S')


def datetime_to_hours(dt, reference_year, date_format='%Y:%m:%d %H'):
    reference_date = datetime(reference_year, 1, 1)
    date = datetime.strptime(dt, date_format)
    difference = date - reference_date
    total_hours = difference.total_seconds() / 3600
    return total_hours


def prepare_future_currents(df, line_dir):
    if 'XL' in df.columns:  # ADCP case
        return df['JUL_SECOND'].values, df['XL'].values
    else:  # PA_ files
        year = int(str(df['Date&Time (GMT+11h)'].values[0])[:4])
        df['Date&Time (GMT+11h)'] = pd.to_datetime(df['Date&Time (GMT+11h)'])
        df['JUL_SECOND'] = (df['Date&Time (GMT+11h)'] - pd.to_datetime(f'{year}-01-01')).dt.total_seconds() - 11 * 3600

        speed = df['Curr. spd (m/s)']
        angle = df['Curr. dir (deg towards)'] / 180 * np.pi
        xl = (speed * np.cos(angle - line_dir * np.pi / 180)).tolist()

        return df['JUL_SECOND'].values, xl
