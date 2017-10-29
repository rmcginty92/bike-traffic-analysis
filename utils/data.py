import os
import boto3
import botocore
import numpy as np
import pandas as pd
from datetime import datetime
import forecastio
import utils.io as io
import utils.auth as auth

from utils.features import expand_datetime_features


def load_features():
    res = io.load_file('feature_file')
    y_nb = res['fremont_bridge_nb']
    y_sb = res['fremont_bridge_sb']
    y = res['y']
    # Remove, date, y_nb and y_sb because they are answers.
    y_cols = ['fremont_bridge_nb', 'fremont_bridge_sb','y']
    feature_cols = res.columns.drop(y_cols)
    return res[feature_cols],y
 

def add_timesteps(X,y=None,timesteps=1):
    #TODO: utilize strided methods in numpy
    # X = (Num samples x num features) --> (Num samples x timesteps x num features)
    X3d = np.column_stack([np.expand_dims(X[timesteps:,:],axis=1)]+[np.expand_dims(X[(timesteps-i):-i,:],axis=1) for i in range(1,timesteps)])
    if y is not None:
        y = y[timesteps:]
        return X3d,y
    return X3d


# ------------------------ #
# Data Retrieval Functions #
# ------------------------ #

def load_data_from_s3(filename=None, s3_resource=None, s3_key=None):
    if s3_resource is None: s3_resource = _create_s3_resource()
    if s3_key is None:
        if isinstance(filename,(str,unicode)):
            s3_key = filename
        else:
            s3_key = io.get_params("s3_key")
    full_filename = resolve_filepath(filename)
    s3_bucket, region = io.get_params("s3_bucket","s3_region")
    try:
        s3_resource.Bucket(s3_bucket).download_file(s3_key, full_filename)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def put_data_in_s3(filename=None, s3_resource=None, s3_key=None):
    if s3_resource is None: s3_resource = _create_s3_resource()
    if s3_key is None:
        if isinstance(filename,(str,unicode)):
            s3_key = filename
        else:
            s3_key = io.get_params("s3_key")
    s3_bucket, region = io.get_params("s3_bucket", "s3_region")
    full_filename = resolve_filepath(filename)
    s3_resource.upload_file(full_filename, s3_bucket, s3_key)



def resolve_filepath(filename):
    # Check if absolute path, then if filename in datapath, finally use default
    if filename is not None and isinstance(filename,str) and os.path.exists(filename):
        return filename
    elif filename is not None and os.path.exists(os.path.join(io.get_path("data_path"),filename)):
        return os.path.join(io.get_path("data_path"), filename)
    else:
        return io.get_filepath("data_file")


def load_data_from_local(filename=None, verbose=True):
    bike_data = None
    full_filename = resolve_filepath(filename)
    try:
        bike_data = io.load_file(full_filename)
    except:
        if verbose: print "ERROR: Bike data not loaded."
    return bike_data


def load_bike_data(filename=None,update=False,save_data=True, verbose=True):
    bike_data = load_data_from_local()
    if update or bike_data is None:
        if verbose: print "Updating..."
        bike_data = pull_bike_data(bike_data=bike_data,save_data=save_data,verbose=verbose)
        bike_data = pull_weather_data(bike_data,filename=filename,save_data=save_data,verbose=verbose)
    # Convert Date into datetime object
    bike_data['date'] = pd.to_datetime(bike_data['date'])
    if verbose: print "Finished."
    return bike_data


def pull_bike_data(bike_data=None,save_data=True, verbose=True):
    database_date_format, date_format, url, dataset_identifier = io.get_params('database_date_format',
                                                                               'date_format','url',
                                                                               'dataset_identifier')
    info = io.load_file("info_file")
    last_pulled = info.get('last_updated',None)
    last_pulled_dt = datetime.strptime(last_pulled, date_format) if last_pulled is not None else datetime(1970,1,1)
    if bike_data is None:
        client = auth.get_soda_client(url)
        json_data = client.get(dataset_identifier=dataset_identifier,content_type='json',limit=50000)
        bike_data = pd.DataFrame(json_data)
        date = pd.to_datetime(bike_data['date'])
        expand_datetime_features(bike_data, date)
        info['last_updated'] = datetime.now().strftime(date_format)
        if verbose: print "New Data Pulled: {0} new rows of data".format(str(bike_data.shape[0]))
    elif datetime.date(last_pulled_dt) < datetime.date(datetime.today()):
        if verbose: print "Last Checked database on {0}, Checking for New Data".format(last_pulled_dt)
        client = auth.get_soda_client(url)
        new_json_data = client.get(dataset_identifier=dataset_identifier,
                                   content_type='json',
                                   limit=50000,
                                   where="date > \"{0}\"".format(bike_data.loc[bike_data.index[-1]]['date']))
        new_bike_data = pd.DataFrame(new_json_data)
        info['last_updated'] = datetime.now().strftime(date_format)
        if 'date' in new_bike_data.columns: # Bedeutet data was pulled from server
            expand_datetime_features(new_bike_data, 'date')
            bike_data = bike_data.append(new_bike_data,ignore_index=True)
            if verbose: print "New data pulled:\n{0} new rows of data\nTotal = {1}".format(str(new_bike_data.shape[0]),str(bike_data.shape[0]))
    else:
        if verbose: print "No new data"

    # TODO: convert numeric columns into numeric types from object types
    bridge_count_numeric = pd.to_numeric(bike_data.loc[:,['fremont_bridge_sb','fremont_bridge_nb']])
    bike_data.loc[:, ['fremont_bridge_sb', 'fremont_bridge_nb']] = bridge_count_numeric
    bike_data['y'] = bike_data['fremont_bridge_nb'].astype(float)+bike_data['fremont_bridge_sb'].astype(float)

    expand_datetime_features(bike_data,'date')
    if save_data:
        io.save_file("data_file",bike_data)
        io.save_file("info_file",info)
    if verbose: print "Bike data is up-to-date."
    return bike_data


def pull_weather_data(bike_data,filename=None,save_data=True, verbose=True):
    lat, lng, date_format = io.get_params('lat', 'lng', 'date_format')
    api_key = io.load_key('forecastio')
    if bike_data['date'].dtype == str:
        try:date = bike_data['date'].apply(lambda x: datetime.strptime(x,date_format))
        except:return bike_data
    else: date = bike_data['date']
    save_freq = 100
    count = 0
    for i,currtime in zip(date[bike_data['hour'] == 0].index,date[bike_data['hour'] == 0]):
        # Each forecast call contains entire day's weather info. Only 1000 free calls a day
        # Do not call if values already present
        if 'apparentTemperature' in bike_data.columns and bike_data['apparentTemperature'].loc[i:i+24].notnull().any():
            if verbose: print "Weather information already pulled from {0}...".format(currtime.date())
            continue
        forecast = forecastio.load_forecast(api_key,lat,lng,time=currtime)
        # ensure no more calculations completed if no data returned; number of calls have been reached
        if forecast is None \
                or (isinstance(forecast,forecastio.models.Forecast)
                    and len(forecast.hourly().data)==0):
            continue
        hourly_data = forecast.hourly().data
        hourly_data_df = pd.concat([pd.DataFrame(hd.d,index=[1]) for hd in hourly_data],ignore_index=True)
        if hourly_data_df.isnull().sum().sum() > 5*hourly_data_df.shape[0]: # Assuming that if 5 cols null, data not good.
            if verbose: print "Failure: Weather info was null".format(currtime.date())
        try:
            bike_data.loc[np.arange(i,i+24),hourly_data_df.columns] = hourly_data_df.values
        except KeyError:
            for col in hourly_data_df.columns:
                bike_data.loc[np.arange(i,i+24),col] = hourly_data_df[col]
        if verbose: print "Weather info successfully scraped for {0}".format(currtime.date())
        count+=1
        if save_data and (count+1) % save_freq == 0:
            if verbose: "Saving  "
            if isinstance(filename,str):
                bike_data.to_csv(filename,index=False)
            else:
                io.save_file('data_file',bike_data,index=False)
        if verbose: print "Finished..."

    if save_data:
        if filename is not None:
            if verbose: print "Saving to {0} file".format(filename)
            io.save_file(io.get_path(['data_path',filename]),bike_data,index=False)
        else:
            if verbose: print "Saving to file specified in cfg.json"
            io.save_file('data_file',bike_data,index=False)
    return bike_data


def _create_s3_resource():
    try:
        ACCESS_KEY = io.load_key("aws.access.key")
        SECRET_KEY = io.load_key("aws.secret.key")
    except:
        raise IOError()
    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    return s3

if __name__=="__main__":
    # d = load_bike_data(update=False,save_data=False)
    # print d
    pass