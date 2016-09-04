from modules import *

def update_df_datetime(df,date):
    df['year'] = date.apply(lambda x: x.year)
    df['month'] = date.apply(lambda x: x.month)
    df['day'] = date.apply(lambda x: x.day)
    df['hour'] = date.apply(lambda x: x.hour)
    df['day_of_week'] = date.apply(lambda x: x.weekday())


def pull_bike_data(bike_data=None,save_data=True,filename='bike_data.csv', verbose=True):
    global date_format, dataset_identifier, info, json_info_file
    last_pulled = info['bike_data']['last_pulled'].split('T')[0]
    if bike_data is None:
         json_data = client.get(dataset_identifier=dataset_identifier,content_type='json',limit=50000)
         N_entries = len(json_data)
         bike_data = pd.DataFrame(json_data)
         date = bike_data['date'].apply(lambda x: datetime.strptime(x,date_format))
         update_df_datetime(bike_data,date)
         info['bike_data']['last_pulled'] = datetime.now().strftime(date_format)
         if verbose: print "New Data Pulled:\n{0} new rows of data".format(str(bike_data.shape[0]))
    elif datetime.date(datetime.strptime(last_pulled,date_format.split('T')[0])) < datetime.date(datetime.today()):
        if verbose: print "Last Checked database on {0}, Checking for New Data".format(last_pulled)
        new_json_data = client.get(dataset_identifier=dataset_identifier,
                                   content_type='json',
                                   limit=50000,
                                   where="date > \"{0}\"".format(bike_data.loc[bike_data.index[-1]]['date']))
        new_bike_data = pd.DataFrame(new_json_data)
        info['bike_data']['last_pulled'] = datetime.now().strftime(date_format)
        if 'date' in new_bike_data.columns: # Bedeutet data was pulled from server
            new_dates = new_bike_data['date'].apply(lambda x: datetime.strptime(x,date_format))
            update_df_datetime(new_bike_data,new_dates)
            bike_data = bike_data.append(new_bike_data)
            if verbose: print "New data pulled:\n{0} new rows of data\nTotal = {1}".format(str(new_bike_data.shape[0]),str(bike_data.shape[0]))
    else:
        if verbose: print "No new data"
    if save_data:
        bike_data.to_csv(filename,index=False)
        with open(json_info_file,'w') as f:
            json.dump(info,f)
    if verbose: print "Bike data is up-to-date."
    return bike_data


def load_bike_data(filename='bike_data.csv',update=False,save_data=True, verbose=True):
    if os.path.exists(filename):
        if verbose: print "Loading File: "+filename
        bike_data = pd.read_csv(filename,index_col=False)
    else:
        if verbose: print "No file named \""+filename+"\"\nPull data from online database using pull_bike_data()..."
        bike_data = pull_bike_data(filename=filename,save_data=save_data,verbose=verbose)
        bike_data = pull_weather_data(bike_data,filename=filename,save_data=save_data,verbose=verbose)
    if update:
        bike_data = pull_bike_data(bike_data=bike_data,save_data=save_data,verbose=verbose)
        bike_data = pull_weather_data(bike_data,filename=filename,save_data=save_data,verbose=verbose)
    return bike_data


def pull_weather_data(bike_data,filename='bike_data.csv',save_data=True, verbose=True):
    global lat,lng, date_format, dataset_identifier, info, json_info_file
    try:
        date = bike_data['date'].apply(lambda x: datetime.strptime(x,date_format))
    except:
        date = bike_data['date'].apply(lambda x: datetime.strptime(x,date_format2))
    save_freq = 100
    count = 0
    for i,currtime in zip(date[bike_data['hour'] == 0].index,date[bike_data['hour'] == 0]):
        # Each forecast call contains entire day's weather info. Only 1000 free calls a day
        # Do not call if values already present
        if 'apparentTemperature' in bike_data.columns and bike_data.loc[i:i+24].isnull().values.sum().sum() < 5*24:
            if verbose: print "Weather information already pulled from {0}...".format(currtime.date())
            continue
        forecast = forecastio.load_forecast(api_key,lat,lng,time=currtime)
        # ensure no more calculations completed if no data returned; number of calls have been reached
        if forecast is None or (isinstance(forecast,forecastio.models.Forecast) and len(forecast.hourly().data)==0): continue
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
            bike_data.to_csv(filename,index=False)
    if verbose: print "Finished..."
    if save_data:
        if verbose: print "Saving to {0} file".format(filename)
        bike_data.to_csv(filename,index=False)
    return bike_data
