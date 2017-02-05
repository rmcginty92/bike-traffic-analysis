import numpy as np
import pandas as pd

import utils.io as io
import utils.gen as gen
import utils.alg as alg

# ------------------------- #
# Preprocessing & Filtering #
# ------------------------- #

def remove_features(features,col_list):
    return features.drop(col_list,axis=1)


def clean_features(feature_set, labels=None, remove_sample_rows=True, remove_feature_cols=True,
                   bad_vals=None, replace_large_vals=False, inf_threshold=None,
                   nan_perc_per_feature_thresh=0.001):
    if bad_vals is None: bad_vals = [np.nan, np.inf, -np.inf]
    i_clean = np.ones(feature_set.shape[0]).astype(bool)
    if replace_large_vals:
        if inf_threshold is None:
            inf_threshold = 1e10 * np.ones(feature_set.shape[1])
        elif isinstance(inf_threshold, (int, float)):
            inf_threshold = inf_threshold * np.ones(feature_set.shape[1])
        else:
            pass  # feed in personalized
        feature_set[feature_set.abs() > inf_threshold] = np.nan
    feature_set = feature_set.replace(bad_vals, np.nan, inplace=False)

    if remove_feature_cols:
        if nan_perc_per_feature_thresh < 1 and nan_perc_per_feature_thresh >= 0:
            nan_perc_per_feature_thresh *= feature_set.loc[i_clean, :].shape[0]
        cols2remove = feature_set.loc[i_clean, :].isnull().sum(0) > int(nan_perc_per_feature_thresh)
        feature_set = remove_features(feature_set, feature_set.columns[cols2remove])
    if remove_sample_rows:
        i_clean = i_clean & feature_set.notnull().all(1).values
    if labels is not None:
        return feature_set.loc[i_clean, :], labels[i_clean]
    else:
        return feature_set.loc[i_clean, :]

def normalize_features(xdf, norm_mask=None, norm_type='max', norm_perc=None, save_norm_scalars=False, eps=1e-20):
    if norm_type == 'robust':
        from sklearn.preprocessing import RobustScaler
        wc = True
        ws = True
        rscaler = RobustScaler(with_centering=wc, with_scaling=ws, quantile_range=(0.25, 0.75))
        nxdf = xdf.copy()
        nxdf.loc[:, :] = rscaler.fit_transform(xdf)
        return nxdf
    elif norm_type == 'standardize':
        from sklearn.preprocessing import scale
        wm = True
        wstd = True
        nxdf = xdf.copy()
        nxdf.loc[:, :] = scale(xdf.values, axis=0, with_mean=wm, with_std=wstd)
        return nxdf
    if norm_type == 'std':
        norm_func = lambda x: np.std(np.abs(x), axis=0)
    elif norm_type == 'mean':
        norm_func = lambda x: np.mean(np.abs(x), axis=0)
    elif norm_type == 'median':
        norm_func = lambda x: np.median(np.abs(x), axis=0)
    elif norm_type in ['quartile', 'perc'] and norm_perc <= 1. and norm_perc > 0:
        norm_func = lambda x: np.percentile(np.abs(x), norm_perc, axis=0)
    else:  # 'max'
        norm_func = lambda x: np.max(np.abs(x), axis=0)
    if norm_mask is None: norm_mask = np.ones(xdf.columns.shape, dtype=bool)
    maxvals = np.ones(xdf.columns.shape)
    maxvals[norm_mask] = norm_func(xdf.loc[:, norm_mask])
    return xdf / (maxvals + eps)


# ------------------ #
# Expanding Features #
# ------------------ #


def expand_datetime_features(df, date):
    if isinstance(date,str):
        date = pd.to_datetime(df[date])
        df['date'] = date
    df['year'] = date.apply(lambda x: x.year)
    df['month'] = date.apply(lambda x: x.month)
    df['day'] = date.apply(lambda x: x.day)
    df['hour'] = date.apply(lambda x: x.hour)
    df['day_of_week'] = date.apply(lambda x: x.weekday())
    df['is_weekend'] = df['day_of_week'] >= 5
    df['day_of_year'] = date.apply(lambda x: x.timetuple().tm_yday)


def expand_features(bike_data):
    # First add hour peaks based on day of week
    bike_data_week = ~bike_data.is_weekend
    # try with np.sqrt and np.log1p
    add_peak_kernels(bike_data,groupby_feature='hour',filtered_bd=bike_data[bike_data_week],colname='weekdayhour'+'_euclid_dist_from',
                     kernel_func=[np.sqrt,np.square],kernel_kwargs=[{},{}])
    bike_data_weekend = bike_data.is_weekend
    add_peak_kernels(bike_data,groupby_feature='hour',filtered_bd=bike_data[bike_data_weekend],colname='weekendhour'+'_euclid_dist_from',
                     peak_lim=1,kernel_func=np.square)
    cols = ['hour','day_of_year','day_of_week','y']
    yr_dist = bike_data.copy()
    yr_dist['y'] = bike_data['y'].rolling(7).mean()
    add_peak_kernels(bike_data,groupby_feature='day_of_year',agg_func=np.max,filtered_bd=yr_dist,
                     colname='day_of_year'+'_euclid_dist_from',kernel_func=np.square,peak_lim=1)
    convert_weather_summary(bike_data)


def convert_weather_summary(bike_data):
    ''' Convert string summary into integer values

    Categories of summary text:

    Cloud Coverage:
        0 - Clear
        1 - Partly Cloudy
        2 - Mostly Cloudy
        3 - Overcast

    Rain:
        0 - Clear
        0 - Dry
        1 - Foggy
        2 - Drizzle
        3 - Light Rain
        4 - Rain
        5 - Heavy Rain

    Day/Night:
        day - 0
        night - 1

    Wind:
        0 - Clear
        1 - Breezy
        2 - Windy
    '''

    def subset_key(x,keys):
        for k  in keys:
            if k in x: return k
        return x

    cloud_conv_dict  = {
        'Clear': 0,
        'Partly Cloudy': 1,
        'Mostly Cloudy': 2,
        'Overcast': 3
    }
    rain_conv_dict = {
        'Clear': 0,
        'Drt': 0,
        'Foggy': 1,
        'Drizzle': 2,
        'Light Rain': 3,
        'Rain': 4,
        'Heavy Rain': 5
    }
    wind_conv_dict = {
        'Clear':0,
        'Breezy':1,
        'Windy':2
    }
    cloud_labels = bike_data['summary'].apply(lambda x: cloud_conv_dict.get(subset_key(str(x),cloud_conv_dict.keys()),0))
    rain_labels = bike_data['summary'].apply(lambda x: rain_conv_dict.get(subset_key(str(x),rain_conv_dict.keys()),0))
    wind_labels = bike_data['summary'].apply(lambda x: wind_conv_dict.get(subset_key(str(x),wind_conv_dict.keys()),0))
    bike_data['cloud_labels'] = cloud_labels
    bike_data['rain_labels'] = rain_labels
    bike_data['wind_labels'] = wind_labels


def find_peaks_of_data(signal,peak_lim=2,trough_lim=2):
    mn_index, mx_index= len(signal),2*len(signal)
    full_signal = np.concatenate((signal,signal,signal))
    peaks, troughs = alg._peakdet(gen.norm(full_signal), delta=0.4)
    peaks_of_interest = peaks[(peaks[:,0]>mn_index) & (peaks[:,0]<=mx_index),:]
    troughs_of_interest = troughs[(troughs[:,0]>mn_index) & (troughs[:,0]<=mx_index),:]
    peaks_of_interest.sort(0); troughs_of_interest.sort(0)
    return peaks_of_interest[-max(peak_lim,1):,0]-mn_index, troughs_of_interest[:max(trough_lim,1),0]-mn_index


def add_peak_kernels(bike_data, groupby_feature, y_feature='y',filtered_bd=None, peaks=None, peak_lim=2,trough_lim=2,
                     colname=None, agg_func=np.sum,kernel_scale=1, kernel_func=np.square,kernel_kwargs={}):
    if filtered_bd is None: filtered_bd = bike_data
    if peaks is None:
        peaks,troughs = find_peaks_of_data(filtered_bd.groupby(groupby_feature).agg(agg_func)[y_feature], peak_lim=peak_lim,trough_lim=trough_lim)
    if colname is None: colname = groupby_feature+'_euclid_dist_from'
    nunique = len(bike_data[groupby_feature].unique())
    if isinstance(kernel_func,list):
        pk_kernel_func,trgh_kernel_func = kernel_func[0],kernel_func[1]
        pk_kernel_kwargs,trgh_kernel_kwargs = kernel_kwargs[0],kernel_kwargs[1]
    else:
        pk_kernel_func,trgh_kernel_func = kernel_func,kernel_func
        pk_kernel_kwargs,trgh_kernel_kwargs = kernel_kwargs,kernel_kwargs
    for peakval in peaks:
        euclid_dist = np.min(np.abs(peakval-[bike_data[groupby_feature]+shift for shift in [-nunique,0,nunique]]),0)
        bike_data[colname+"_pk_"+str(int(peakval))] = pk_kernel_func(kernel_scale*euclid_dist,**pk_kernel_kwargs)
    for trough in troughs:
        euclid_dist = np.min(np.abs(trough-[bike_data[groupby_feature]+shift for shift in [-nunique,0,nunique]]),0)
        bike_data[colname+"_trgh_"+str(int(trough))] = trgh_kernel_func(kernel_scale*euclid_dist,**trgh_kernel_kwargs)



