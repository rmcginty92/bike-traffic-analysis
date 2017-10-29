import utils.data
import utils.alg
import utils.gen
import utils.features
import pandas as pd
import numpy as np

Xfull,yfull = utils.data.load_features()
Xfull = utils.features.clean_features(Xfull)

# Main features of interst
weather_features = [
    'apparentTemperature',
    'dewPoint',
    'humidity',
    'precipIntensity',
    'precipProbability',
    'pressure',
    'temperature',
    'visibility',
    'windBearing',
    'windSpeed',
    'cloud_labels',
    'rain_labels',
    'wind_labels'
]



feature_cols = Xfull.columns.tolist()
is_weekend = Xfull['is_weekend']

# Split weekday, weekend data
X1,y1 = Xfull.loc[~is_weekend,:],yfull[~is_weekend]
X2,y2 = Xfull.loc[is_weekend,:],yfull[is_weekend]

# Select week/weekend
X, y = X1, y1

#%% Evaluating Ridership at peaks

# Setting variables
morning_hours = np.array([7,8,9])
evening_hours = np.array([16,17,18])
x_agg = np.mean
y_agg = np.sum
y_lag = 1 # hour
#%%

Xmorning_rng = X[X.hour.isin(morning_hours-y_lag)]
Xevening_rng = X[X.hour.isin(evening_hours-y_lag)]

ymorning_rng = y[X.hour.isin(morning_hours)]
yevening_rng = y[X.hour.isin(evening_hours)]
morning_date = pd.to_datetime(X[X.hour.isin(morning_hours)][['year','month','day']])
evening_date = pd.to_datetime(X[X.hour.isin(evening_hours)][['year','month','day']])




Xmorning = Xmorning_rng.groupby(['year','month','day']).mean()
Xevening = Xevening_rng.groupby(['year','month','day']).mean()

ymorning = ymorning_rng.groupby(morning_date).sum()
yevening = yevening_rng.groupby(evening_date).sum()


#%%


