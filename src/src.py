import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import sklearn as skl
import matplotlib.pyplot as plt
plt.close('all')
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

df = pd.read_csv('../data/raw/nyc_taxi.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['isAnomaly'] = 0
df.loc[df.timestamp == '2014-11-01 19:00:00', 'isAnomaly'] = 1
df.loc[df.timestamp == '2014-11-27 15:30:00', 'isAnomaly'] = 1
df.loc[df.timestamp == '2014-12-25 15:00:00', 'isAnomaly'] = 1
df.loc[df.timestamp == '2015-01-01 01:00:00', 'isAnomaly'] = 1
df.loc[df.timestamp == '2015-01-27 00:00:00', 'isAnomaly'] = 1

df['isNight'] = ((df.timestamp.dt.hour >= 19) | (df.timestamp.dt.hour < 5)).astype(int)
df['isWeekend'] = (df.timestamp.dt.dayofweek >= 5).astype(int)

holidays = calendar().holidays(start=df.timestamp.min(), end=df.timestamp.max())
df['isMajorHoliday'] = df['timestamp'].dt.date.astype('datetime64').isin(holidays).astype(int)

df['timeOfDay'] = (df.timestamp.dt.hour + df.timestamp.dt.minute / 60) / 23.5

df.to_csv('../data/processed/nyc_taxi_transformed.csv')

df.plot(x='timestamp', y='value', title='Total NYC Taxi Passenger Counts per Thirty Minutes vs. Timestamp')

df.head(24*2*7).plot(x='timestamp', y='value', title='One Week of Total NYC Taxi Passenger Counts per Thirty Minutes vs. Timestamp')

df[df.timestamp.dt.date == pd.Timestamp('2014-11-01')].plot(x='timestamp', y='value', title='NYC Marathon Anomaly Occuring at 19:00 Hours')

df[df.timestamp.dt.date == pd.Timestamp('2014-11-27')].plot(x='timestamp', y='value', title='Thanksgiving Anomaly Occuring at 15:30 Hours')

df[df.timestamp.dt.date == pd.Timestamp('2014-12-25')].plot(x='timestamp', y='value', title='Christmas Anomaly Occuring at 15:00Hours')

df[df.timestamp.dt.date == pd.Timestamp('2015-01-01')].plot(x='timestamp', y='value', title='New Year\'s Day Anomaly Occuring at 01:00 Hours')

df[df.timestamp.dt.date == pd.Timestamp('2015-01-27')].plot(x='timestamp', y='value', title='Blizzard Anomaly Occuring at 00:00 Hours')

X_value = df[['value']].values
X_time = df[['value', 'timeOfDay']].values
X_all = df[['value', 'timeOfDay', 'isNight', 'isWeekend', 'isMajorHoliday']].values

V_IF = IsolationForest().fit_predict(X_value)
df['V_IF_isAnomaly'] = pd.Series(V_IF)

V_LOF = LocalOutlierFactor().fit_predict(X_value)
df['V_LOF_isAnomaly'] = pd.Series(V_LOF)

V_OCSVM = OneClassSVM().fit_predict(X_value)
df['V_OCSVM_isAnomaly'] = pd.Series(V_OCSVM)

T_IF = IsolationForest().fit_predict(X_time)
df['T_IF_isAnomaly'] = pd.Series(T_IF)

T_LOF = LocalOutlierFactor().fit_predict(X_time)
df['T_LOF_isAnomaly'] = pd.Series(T_LOF)

T_OCSVM = OneClassSVM().fit_predict(X_time)
df['T_OCSVM_isAnomaly'] = pd.Series(T_OCSVM)

A_IF = IsolationForest().fit_predict(X_all)
df['A_IF_isAnomaly'] = pd.Series(A_IF)

A_LOF = LocalOutlierFactor().fit_predict(X_all)
df['A_LOF_isAnomaly'] = pd.Series(A_LOF)

A_OCSVM = OneClassSVM().fit_predict(X_all)
df['A_OCSVM_isAnomaly'] = pd.Series(A_OCSVM)

df['value'] = (df['value'] - 8) / 39189

X_time = df[['value', 'timeOfDay']].values

T_LOF = LocalOutlierFactor().fit_predict(X_time)
df['T_LOF_V2_isAnomaly'] = pd.Series(T_LOF)

X_all = df[['value', 'timeOfDay', 'isNight', 'isWeekend', 'isMajorHoliday']].values

A_LOF = LocalOutlierFactor().fit_predict(X_all)
df['A_LOF_V2_isAnomaly'] = pd.Series(A_LOF)

print(df)