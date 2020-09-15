import MySQLdb as sql


def connect_to_data_base():
	try:
		db = sql.connect(host='164.125.154.217', user='root', passwd='2848', db='locs_keywords',
						 port=3306, charset='utf8')
	except:
		raise ConnectionError('Could not connect to database server, check internet connection and database detail..')
	return db


# import numpy as np
# import pandas as pd
# import requests
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
#
#
# def _pad_input(x, length, pad_char=0):
#     return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=pad_char)
#
#
#
# def prepare_data(data_path: str):
#     data_path='data/processed/train.txt'
#     data= None
#
#     max_length = 0
#     with open(data_path, mode='r', encoding='utf-8') as f:
#         for line in f:
#             data = line.split('|')
#             longest_sequence = max(data, key=len)
#             temp = longest_sequence.split('-')[0]
#             temp = temp.strip('[]').replace(' ', '')
#             max_length = np.array([int(a_) for a_ in temp.split(',')]).shape[0]
#
#     inputs_array = []
#     labels_array = []
#     for row in data[:-1]:
#         in_ = row.split('-')[0]
#         in_=in_.strip('[]').replace(' ','')
#         in_ = np.array([int(a_) for a_ in in_.split(',')])
#         # inputs = _pad_input(in_, max_length, pad_char=0)
#         labels = int(row.split('-')[1])
#         inputs_array.append(inputs)
#         labels_array.append(labels)
#
#     print(len(inputs_array))
#     print(len(labels_array))
#
#
#     return inputs, labels
#
# data_path = 'data/processed/train.txt'
#
# inputs, labels = prepare_data(data_path)
#
#
# def get_shapes(x):
#     """Return list of dims, statically where possible."""
#     x = tf.convert_to_tensor(x)
#     # If unknown rank, return dynamic shape
#     if x.get_shape().dims is None:
#         return tf.shape(x)
#
#     static = x.get_shape().as_list()
#     shape = tf.shape(x)
#
#     shapes = []
#     for i in range(len(static)):
#         dim = static[i]
#         if dim is None:
#             dim = shape[i]
#         shapes.append(dim)
#     return shapes
#
#
#
#
# def login(config):
#     ntels_login_url = config.ntels_host + config.login_url
#     login_params = dict(
#         user_id=config.ntels_username,
#         user_pw=config.ntels_password
#     )
#     with requests.Session() as login_sess:
#         try:
#             login_sess.post(ntels_login_url, login_params, verify=False, allow_redirects=True, timeout=90)
#             return login_sess
#         except:
#             raise ConnectionError('Could not login to server, check server configuration..')
#
#
# def get_data_from_HTTPS_request(url, login_sess):
#     '''
#     get json data format from HTTP API
#     :param url: http address
#     :param params: parameters
#     :return: json data format
#     '''
#     ### login first
#     try:
#         response = login_sess.get(url, verify=False, timeout=90)
#         return response.json()
#     except Exception as e:
#         raise Exception('Could not get data from {} with GET request, check server API configuration'.format(url))
#
#
# def get_data_from_HTTPS_post_request(url, login_sess, params):
#     '''
#         get json data format from HTTP API
#         :param url: http address
#         :param params: parameters
#         :return: json data format
#         '''
#     ### login first
#     try:
#         response = login_sess.post(url, data=params, verify=False, timeout=90)
#         return response.json()
#     except:
#         raise ConnectionError(
#             'Could not get data from {} with POST request, check server API configuration'.format(url))
#
#
#
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.sum(np.abs((y_true - y_pred) / y_true)) / len(y_pred)
#
#
# def json_to_dataframe(path, col_name='Power', orient='index'):
#     import json
#     with open(path) as in_stream:
#         data_dict = json.load(in_stream)
#     data = pd.DataFrame.from_dict(data_dict, orient=orient, columns=[col_name])
#     return data
#
#
# ##### export dataframe to json format  = {index : value}
# def dataframe_to_dict(data):
#     data_dict = {}
#     for idx, row in data.iterrows():
#         data_dict[str(idx)] = row[0]
#     return data_dict
#
#
# ## export dictionary to json file format
# def dict_to_json(folder, filename, dict):
#     '''
#     dict must be a 1 level dictionary )not contains list
#     :param folder:
#     :param filename:
#     :param dict:
#     :return:
#     '''
#     import json
#     import os
#     ## create 'analyzed' folder (if neccessary)
#     if folder != '':
#         os.makedirs(folder, exist_ok=True)
#     with open(os.path.join(folder, filename), 'w') as out_stream:
#         json.dump(dict, out_stream)
#
#
# ### export dataframe to csv file format
# def dataframe_to_csv(folder, filename, dataframe):
#     import os
#     if folder != '':
#         os.makedirs(folder, exist_ok=True)
#         path = os.path.join(folder, filename)
#     else:
#         path = filename
#     try:
#         dataframe.to_csv(path)
#     except:
#         raise IOError('Cannot create file, check permission..')
#
#
# def loadData(path, idx_col=0, freq='H', formula='mean'):
#     '''
#     load data from file in path
#     :param path: file path
#     :param idx_col: index of indexing column
#     :return: dataframe + building name
#     '''
#     try:
#         if path.endswith('.csv'):
#             data = pd.read_csv(path)
#             building = path.split('\\')[-1].split('/')[-1].replace('.csv', '')
#         elif path.endswith('.xlsx') or path.endswith('.xls'):
#             data = pd.read_excel(path)
#             building = path.split('\\')[-1].split('/')[-1].replace('.xlsx', '').replace('.xls', '')
#         elif path.endswith('.json'):
#             data = json_to_dataframe(path)
#             building = path.split('\\')[-1].split('/')[-1].replace('.json', '')
#         else:
#             print('Not supported file format..')
#             return
#     except:
#         raise Exception('Cannot read files format from input folder..')
#     ## set data index
#     datetimeColName = data.columns[idx_col]
#     data[datetimeColName] = pd.to_datetime(data[datetimeColName])
#     data.set_index(datetimeColName, inplace=True)
#     ## sort data by index
#     data = data.sort_index(axis=0)
#     ### checking for non-numertic value and convert using interpolate function
#     for i in data.columns:
#         data[i] = pd.to_numeric(data[i], errors='coerce').interpolate(method='quadratic')
#     ##### fill missing data with interpolate value
#     # (suppose that 15 minutes is minimum data collected interval.
#     # 1 minutes make function performance slow down)
#     # if data is collected using smaller interval, decrease frequence param
#     data = data.resample('15T').interpolate()
#     data = data.reindex(pd.date_range(data.index[0], data.index[-1], freq=freq)).fillna(method='bfill')
#     ### resample data to freq regulation
#     if formula == 'sum':  ## resampling using sum function
#         ### resample data to fill missing value. then reindex to 15 frequence
#         data = data.resample(freq).sum()
#     elif formula == 'mean':  ## resampling using mean() function
#         data = data.resample(freq).mean()
#     return data, building
#
#
#
# def scale_to_0and1(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))  # MinMax Scaler
#     data = scaler.fit_transform(data)  # input: ndarray type data
#     return (data, scaler)
#
#
# # split into train and test sets
# def splitData(data, trainPercent=0.8):
#     train_size = int(len(data) * trainPercent)
#     train_data, test_data = data.iloc[0:train_size, ], data.iloc[train_size:, ]
#     print("\n", "train length:", len(train_data), "\n", "test length:", len(test_data))
#     return (train_data, test_data)
#
#
# def get_data_filenames_from_folder(path, ext='.csv'):
#     import os
#     try:
#         filenames = [os.path.join(path, file) for file in os.listdir(path) if
#                      os.path.isdir(os.path.join(path, file)) == False]
#         filenames = sorted(filenames)
#         data_files = []
#         for file in filenames:
#             if file.endswith(ext):
#                 data_files.append(file)
#     except FileNotFoundError:
#         raise FileNotFoundError('Cannot locate the specific folder..')
#     return data_files
#
#
# def get_data_by_time(energy_data, start_time=None, end_time=None, freq='H'):
#     if start_time is None and end_time is None:
#         start_time = energy_data.index[0]
#         end_time = energy_data.index[-1]
#         try:
#             start_time, end_time = pd.to_datetime(start_time), pd.to_datetime(end_time)
#             start_time, end_time = str(start_time.date()), str(end_time.date())
#         except ValueError as e:
#             raise (e)
#     else:
#         ## if one of start or end time is None
#         start_time = end_time if start_time is None else start_time
#         end_time = start_time if end_time is None else end_time
#         start_time = pd.to_datetime(start_time)
#         end_time = pd.to_datetime(end_time)
#         ### if user input time with format YYYY/MM/DD --> set time for start and end time
#         if start_time.time().hour == start_time.time().minute == 0:
#             if freq == 'H':
#                 start_time = start_time.replace(hour=0)
#                 end_time = end_time.replace(hour=23)
#             elif freq == '15T' or freq == '15m':
#                 start_time = start_time.replace(hour=0, minute=0)
#                 end_time = end_time.replace(hour=23, minute=45)
#         else:  ## if user input time with format YYYY/MM/DD HH:MM --> set time for end time corresponding to start time
#             if freq == 'H':
#                 start_time = start_time.replace(minute=0)
#                 end_time = end_time.replace(day=start_time.date().day + 1, hour=start_time.time().hour,
#                                             minute=start_time.time().minute)
#             elif freq == '15T' or freq == '15m':
#                 start_time = start_time.replace(minute=start_time.time().minute - (
#                         start_time.time().minute % 15))  ### round up minute to 15 minute freq
#                 end_time = end_time.replace(day=start_time.date().day + 1, hour=start_time.time().hour,
#                                             minute=start_time.time().minute - (
#                                                     start_time.time().minute % 15))  ### round up minute to 15 minute freq
#     if start_time > end_time:
#         raise ValueError('Start time must be a time before end time..')
#     elif start_time <= end_time:
#         if start_time not in energy_data.index or end_time not in energy_data.index:
#             print('start time {}'.format(start_time))
#             print('end_time  {}'.format(end_time))
#             print(energy_data.index)
#             raise ValueError('Time range exceed data range, please choose different time range..')
#         else:
#             period_data = energy_data[start_time:end_time]
#             return period_data
#
#
# def get_time_by_month(month, year, freq='H'):
#     import datetime
#     today = datetime.date.today()
#     if month is None or int(month) > 12 or int(month) < 1:
#         raise Exception('Unreconized month, make sure month is integer from 1-12..')
#     elif year is None or int(year) < 1970 or int(year) > datetime.date.today().year:
#         raise Exception('Unreconized year, make sure passed year is integer from 1970 to this year..')
#     else:
#         ### determine month first and last day
#         if year == today.year:
#             if month == today.month:
#                 start_time = today.replace(day=1)
#                 start_time = pd.to_datetime(start_time)
#                 end_time = pd.datetime.now()
#                 if freq == 'H':
#                     end_time = end_time.replace(minute=0, second=0, microsecond=0)
#                 elif freq == '15T' or freq == '15m':
#                     end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
#                                                 microsecond=0)
#                 end_time = pd.to_datetime(end_time)
#                 return start_time, end_time
#             elif month < today.month:
#                 latter_month_first_day = today.replace(month=month + 1, day=1)
#                 month_last_day = latter_month_first_day - datetime.timedelta(days=1)
#                 month_first_day = month_last_day.replace(day=1)
#                 ## convert date to date time
#                 start_time = pd.to_datetime(month_first_day)
#                 end_time = pd.to_datetime(month_last_day)
#                 ## edit end_time due to frequence
#                 if freq == 'H':
#                     end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
#                 elif freq == '15T' or freq == '15m':
#                     end_time = end_time.replace(hour=23, minute=45,
#                                                 second=0, microsecond=0)
#                 return start_time, end_time
#             else:
#                 raise ValueError('Month is not valid..')
#         else:
#             if month == 12:
#                 latter_year_first_day = today.replace(year=year + 1, month=1, day=1)
#                 last_day = latter_year_first_day - datetime.timedelta(days=1)
#                 first_day = last_day.replace(day=1)
#                 ## convert date to date time
#                 start_time = pd.to_datetime(first_day)
#                 end_time = pd.to_datetime(last_day)
#                 ## edit end_time due to frequence
#                 if freq == 'H':
#                     end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
#                 elif freq == '15T' or freq == '15m':
#                     end_time = end_time.replace(hour=23, minute=45,
#                                                 second=0, microsecond=0)
#                 return start_time, end_time
#             else:
#                 latter_month_first_day = today.replace(year=year, month=month + 1, day=1)
#                 month_last_day = latter_month_first_day - datetime.timedelta(days=1)
#                 month_first_day = month_last_day.replace(day=1)
#                 ## convert date to date time
#                 start_time = pd.to_datetime(month_first_day)
#                 end_time = pd.to_datetime(month_last_day)
#                 ## edit end_time due to frequence
#                 if freq == 'H':
#                     end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
#                 elif freq == '15T' or freq == '15m':
#                     end_time = end_time.replace(hour=23, minute=45,
#                                                 second=0, microsecond=0)
#                 return start_time, end_time
#
#
# def get_time_by_keyword(keyword='today', freq='H'):
#     # keyword in ['today', 'yesterday', 'this week', 'last week', 'this month', 'last month', 'this year', 'last year']
#     import datetime
#     keyword = keyword.lower().strip()
#     assert keyword in ['tomorow', 'today', 'yesterday', 'this week', 'last week', 'this month', 'last month',
#                        'this year',
#                        'last year']
#     if keyword == 'today':
#         today = pd.datetime.today().date()
#         start_time = pd.to_datetime(today)
#         end_time = pd.datetime.now()
#         if freq == 'H':
#             end_time = end_time.replace(minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
#                                         microsecond=0)
#         end_time = pd.to_datetime(end_time)
#         return start_time, end_time
#     elif keyword == 'tomorow':
#         tomorow = datetime.date.today() + datetime.timedelta(days=1)
#         start_time = pd.to_datetime(tomorow)
#         end_time = start_time
#         if freq == 'H':
#             end_time = start_time.replace(hour=23, minute=0, second=0, microsecond=0)
#         else:
#             end_time = start_time.replace(hour=23, minute=45,
#                                           second=0, microsecond=0)
#         return start_time, end_time
#     elif keyword == 'yesterday':
#         yesterday = datetime.date.today() - datetime.timedelta(days=1)
#         start_time = pd.to_datetime(yesterday)
#         end_time = start_time
#         if freq == 'H':
#             end_time = start_time.replace(hour=23, minute=0, second=0, microsecond=0)
#         else:
#             end_time = start_time.replace(hour=23, minute=45,
#                                           second=0, microsecond=0)
#         return start_time, end_time
#     elif keyword == 'this week':
#         today = datetime.date.today()
#         start_time = today - datetime.timedelta(days=today.weekday())
#         start_time = pd.to_datetime(start_time)
#         end_time = pd.datetime.now()
#         if freq == 'H':
#             end_time = end_time.replace(minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
#                                         microsecond=0)
#         end_time = pd.to_datetime(end_time)
#         return start_time, end_time
#     elif keyword == 'last week':
#         today = datetime.date.today()
#         start_time = today - datetime.timedelta(days=today.weekday() + 7)
#         start_time = pd.to_datetime(start_time)
#         end_time = start_time + datetime.timedelta(days=6)
#         if freq == 'H':
#             end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(hour=23, minute=45,
#                                         second=0, microsecond=0)
#         end_time = pd.to_datetime(end_time)
#         return start_time, end_time
#     elif keyword == 'this month':
#         today = datetime.date.today()
#         this_month_first_day = today.replace(day=1)
#         start_time = pd.to_datetime(this_month_first_day)
#         end_time = pd.datetime.now()
#         if freq == 'H':
#             end_time = end_time.replace(minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
#                                         microsecond=0)
#         end_time = pd.to_datetime(end_time)
#         return start_time, end_time
#     elif keyword == 'last month':
#         today = datetime.date.today()
#         this_month_first_day = today.replace(day=1)
#         last_month_last_day = this_month_first_day - datetime.timedelta(days=1)
#         last_month_first_day = last_month_last_day.replace(day=1)
#         start_time, end_time = last_month_first_day, last_month_last_day
#         start_time = pd.to_datetime(start_time)
#         end_time = pd.to_datetime(end_time)
#         if freq == 'H':
#             end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(hour=23, minute=45,
#                                         second=0, microsecond=0)
#         return start_time, end_time
#     elif keyword == 'this year':
#         today = datetime.date.today()
#         this_year_first_day = today.replace(month=1, day=1)
#         start_time = pd.to_datetime(this_year_first_day)
#         end_time = pd.datetime.now()
#         if freq == 'H':
#             end_time = end_time.replace(minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(minute=end_time.time().minute - (end_time.time().minute % 15), second=0,
#                                         microsecond=0)
#         end_time = pd.to_datetime(end_time)
#         return start_time, end_time
#     elif keyword == 'last year':
#         today = datetime.date.today()
#         last_year_first_day = today.replace(year=(today.year - 1), month=1, day=1)
#         last_year_last_day = today.replace(year=(today.year - 1), month=12, day=31)
#         start_time = last_year_first_day
#         end_time = last_year_last_day
#         start_time = pd.to_datetime(start_time)
#         end_time = pd.to_datetime(end_time)
#         if freq == 'H':
#             end_time = end_time.replace(hour=23, minute=0, second=0, microsecond=0)
#         elif freq == '15T' or freq == '15m':
#             end_time = end_time.replace(hour=23, minute=45,
#                                         second=0, microsecond=0)
#         return start_time, end_time
#     else:
#         print('Unknown keyword..')
#         return
#
#
# def reshapeForLSTM(data, time_steps=None):
#     """
#     :param data: intput data
#     :param time_steps: time steps after
#     :return: reshaped data for LSTM
#     """
#     """
#     The LSTM network expects the input data (X)
#     to be provided with
#     a specific array structure in the form of:
#     [samples, time steps, features].
#     """
#     if time_steps is None:
#         print("please denote 'time_steps'...!")
#         return (None)
#     else:
#         data_reshaped = np.reshape(data, (data.shape[0], time_steps, 1))
#     return (data_reshaped)
#
#
# # --- create dataset with window size --- #
# def sequentialize(scaled_inputData, inputData_index, window_size=None, to_ndarray=False):
#     if window_size is None:
#         print("\n", "please use 'window_size'...!")
#         return (None)
#     elif isinstance(window_size, int):
#         # change type to use 'shift' of pd.DataFrame
#         scaled_inputData = pd.DataFrame(scaled_inputData, columns=["value"], index=inputData_index)
#         # dataframe which is shifted as many as window size
#         for idx in range(1, window_size + 1):
#             scaled_inputData["before_{}".format(idx)] = scaled_inputData["value"].shift(idx)
#         # drop na
#         inputSequence = scaled_inputData.dropna().drop('value', axis=1)
#         output = scaled_inputData.dropna()[['value']]
#
#         if to_ndarray is False:
#             return (inputSequence, output)
#         else:
#             inputSequence = inputSequence.values
#
#             output = output.values
#             return (inputSequence, output)
#
#
#
#
# def get_hourly_temp_by_date(date):
#     import requests
#     import urllib.parse as urlparser
#     import datetime
#     next_date = pd.to_datetime(date) + datetime.timedelta(days=1)
#     next_date = str(next_date.date()).replace('-', '').strip()
#     date = str(date).replace('-', '').strip()
#     url = 'https://data.kma.go.kr/apiData/getData'
#     params = dict(
#         type='json',
#         dataCd='ASOS',
#         dateCd='HR',
#         startDt=date,
#         startHh='00',
#         endDt=next_date,
#         endHh='23',
#         stnIds='119',
#         schListCnt='24',
#         pageIndex='1',
#         apiKey='3zrCi0JNWnE%2BmdIkxKiH/FVTXyU4aYXRinkR21ktFNfhPh9cZtyBbMJhnJjqwNjv',
#     )
#     params_ = urlparser.unquote(urlparser.urlencode(params,
#                                                     doseq=True))  ## join params into url style (unquote to decode ASCII chars back to normal chars)
#     url_ = url + '?' + params_  ### generate full url (cannot get data from separate url and params, so we join them)
#     response = requests.get(url_, verify=False, allow_redirects=True, timeout=90)
#     json = response.json()
#     data = json[3]['info']
#     temps = []
#     times = []
#     for item in data:
#         try:
#             temps.append(item['TA'])  ### get temperature of each hour from 'TA' variable
#             times.append(pd.to_datetime(item['TM']))  ### get time index from 'TM' variable
#         except:
#             pass
#     temps = pd.DataFrame(temps, columns=['Temperature'], index=times)
#     return temps  # int(round(float(np.mean(temps)),0))###
#
#
# def add_hourly_temp_column(data):
#     import time
#     import copy
#     freq = str(data.index.freq)
#     ## make a copy of original data
#     data_ = copy.deepcopy(data)
#
#     ## because the API only support 1 hour frequency temperature, transform  15 minutes frequency to 1 hour frequency
#     if freq == '<15 * Minutes>':
#         data_ = data_.resample('H').sum()
#         ## extract date
#         dates = [str(date.date()) for date in data_.index]
#         dates = sorted(set(dates))
#         ### create empty dataframe to save temperature data
#         temps_df = pd.DataFrame(columns=['Temperature'])
#         ### for counting down  variable
#         i = len(dates)
#         for date in dates:
#             # print('Adding hourly temperature from API. Remaining days: {}'.format(i))  ## print progress for controling
#             i -= 1
#             time.sleep(1)  ### kma's API does not allow getting data too fast.
#             ### get hourly temperature of each date
#             hourly_temp = get_hourly_temp_by_date(date)
#             ## add to total dataframe
#             temps_df = pd.concat([temps_df, hourly_temp], axis=0)
#         ### add temperature data as a column to original dataframe
#         new_data = pd.concat([data_, temps_df], axis=1)
#
#         ### reindex back to 15 minutes frequency
#         new_data_ = new_data.reindex(pd.date_range(new_data.index[0], data.index[-1], freq='15T')).resample(
#             '15T').interpolate().dropna()
#         ### cut out redundant parts
#         new_data_ = new_data_.loc[data.index]
#         ## append temperature column to original data (we do not use interpolated data because we want to keep data as close to reality as possible)
#         new_data__ = pd.concat([data, new_data_['Temperature']], axis=1)
#         return new_data__
#     else:
#         ## extract dates
#         dates = [str(date.date()) for date in data.index]
#         dates = sorted(set(dates))
#         ### create empty dataframe to save temperature data
#         temps_df = pd.DataFrame(columns=['Temperature'])
#         ### for counting down  variable
#         i = len(dates)
#         for date in dates:
#             i -= 1
#             time.sleep(1)  ### kma's API does not allow getting data too fast.
#             ### get hourly temperature of each date
#             hourly_temp = get_hourly_temp_by_date(date)
#             ## add to total dataframe
#             temps_df = pd.concat([temps_df, hourly_temp], axis=0)
#         ### add temperature data as a column to original dataframe
#         new_data = pd.concat([data, temps_df], axis=1)
#         return new_data
#
#
# def add_hourly_temp_column_from_csv(data):
#     ## get data frequency
#     freq = str(data.index.freq)
#     ### get temperature info from file
#     temperatures = pd.read_csv('temperature.csv')
#     ## convert index column to datetime and setindex
#     temperatures[temperatures.columns[0]] = pd.to_datetime(temperatures[temperatures.columns[0]])
#     temperatures.set_index(temperatures.columns[0], inplace=True)
#     temperatures = temperatures.drop_duplicates(subset=temperatures.columns[0], keep='last')
#     if freq == '<15 * Minutes>':
#         ## transform temperature data to 15 minutes freq using interpolate
#         temperatures = temperatures.resample('15T').interpolate()
#         temperatures = temperatures.reindex(pd.date_range(data.index[0], data.index[-1], freq='15T'))
#
#     ### append temperature to data, keep only temperature that match with data's time
#     data_with_temperature = data.join(temperatures, how='left')
#     return data_with_temperature
#
#
# def get_predicted_consumption_from_API(input, building_id):
#     import requests
#     import pandas as pd
#     # building_id='B0002'
#
#     # prediction_API_url = 'http://192.168.13.36:1280/prediction'
#     prediction_API_url = 'http://210.219.151.163:1280/prediction'
#     headers = {
#         'Content-Type': 'application/json',
#         'cache-control': 'no-cache',
#         'Postman-Token': '94ea6528-06d7-46e7-a7c2-5a38c198d1f8'
#     }
#     input_ = str(input).replace('\'', r'"').replace(' ', '')
#     params = "{\"bld_id\" : \"" + building_id + "\",\"token\": \"D6099EC547FAC794B34542A82B12A12C586A3351A2892414CB175179787A894B\",\"data\" : " + input_ + ",\"timestamp\" : \"2018-11-21 09:42:46.502277\"}"
#     try:
#         response = requests.post(prediction_API_url, data=params, headers=headers, timeout=30)
#     except ConnectionError:
#         print(
#             ConnectionError('Could not connect to API server, check server configuration..'.format(prediction_API_url)))
#         return pd.DataFrame(columns=['Power'])
#     ## todo: catch more error here "expecting value: line 1 column 1 (char 0)"
#     try:
#         result = response.json()['prediction_result']
#         result_df = pd.DataFrame(result, columns=['Time', 'Power'])
#         result_df.set_index('Time', inplace=True)
#     except Exception as e:
#         print(Exception(
#             'Error: {} occurred when getting data from {}, check input parameters..'.format(response.content,
#                                                                                             prediction_API_url)))
#         return pd.DataFrame(columns=['Power'])
#     return result_df
