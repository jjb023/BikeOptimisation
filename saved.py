import pandas as pd
import numpy as np

def convert_minutes_to_hours(minutes):
    hours = str(minutes // 60)
    minutes = str(minutes % 60)
    if len(hours) == 1:
        hours = f'0{hours}'
    if len(minutes) == 1:
        minutes = f'0{minutes}'

    return f'{hours}:{minutes}'

class RnnInput:

    def __init__(self):

        df = pd.read_csv('299JourneyDataExtract05Jan2022-11Jan2022.csv')
        df.dropna(how='any', axis=0, inplace=True)

        start_stations = [station for station in df['StartStation Id']]
        end_stations = [station for station in df['EndStation Id']]

        stations = sorted(set(start_stations + end_stations))

        self.stations = list(map(lambda x: int(x), stations))

        self.timeperiod_end = {convert_minutes_to_hours(x): [] for x in range(0, 1440, 30)}
        self.timeperiod_start = {convert_minutes_to_hours(x): [] for x in range(0, 1440, 30)}
        self.find_time_period_hours_end = [convert_minutes_to_hours(x).split(':')[0] for x in range(0, 1440, 30)]

        # New dataframe
        cols = ['Bikes In', 'Bikes Out', 'Current Bike Count', 'Timestep', 'Date']
        array = np.zeros((len(self.stations), len(cols)), int)
        self.RNN_INPUT = pd.DataFrame(array, columns=cols, index=self.stations)

        for index in df.index:

            sample = df.loc[index]
            sample_end_hour = sample['End Date'].split(' ')[-1].split(':')[0]
            sample_end_minute = sample['End Date'].split(' ')[-1].split(':')[-1]

            sample_start_hour = sample['Start Date'].split(' ')[-1].split(':')[0]
            sample_start_minute = sample['Start Date'].split(' ')[-1].split(':')[-1]

            if int(sample_start_minute) < 30:
                key = f'{sample_start_hour}:30'
            else:
                key = f'{sample_start_hour}:00'

            self.timeperiod_start[key].append(sample)

            if int(sample_end_minute) < 30:
                key = f'{sample_end_hour}:30'
            else:
                key = f'{sample_end_hour}:00'
            self.timeperiod_end[key].append(sample)

        self.current_time_step = '00:00'

    def bikes_in(self):

        dataframe_end = pd.DataFrame(self.timeperiod_end[self.current_time_step])

        for index in dataframe_end.index:
            sample = dataframe_end.loc[index]
            start_station = sample['EndStation Id']
            self.RNN_INPUT.loc[start_station, 'Bikes In'] += 1

    def bikes_out(self):

        dataframe_start = pd.DataFrame(self.timeperiod_start[self.current_time_step])

        for index in dataframe_start.index:
            sample = dataframe_start.loc[index]
            start_station = sample['StartStation Id']
            self.RNN_INPUT.loc[start_station, 'Bikes Out'] += 1


    def update_current_bike_count(self):

        for index in self.RNN_INPUT.index:
            sample = self.RNN_INPUT.loc[index]
            bikes_in, bikes_out = sample['Bikes In'], sample['Bikes Out']
            difference = bikes_in - bikes_out
            self.RNN_INPUT.loc[index, 'Current Bike Count'] += difference

        print(self.RNN_INPUT)

    def time_step(self):

        time_step_list = list(self.timeperiod_end.keys())
        time_step = time_step_list.index(self.current_time_step)
        self.RNN_INPUT['Timestep'] = time_step
        print(self.RNN_INPUT)


    def date(self):

        for index in self.dataframe_end.index:
            sample = self.RNN_INPUT.loc[index]
            date = sample['Start Date']
            print(date)




instance = RnnInput()
instance.date()
