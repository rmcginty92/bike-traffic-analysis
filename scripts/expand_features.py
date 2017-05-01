import utils.io as io
import utils.data as data
import utils.features as features


def main():
    df = data.load_bike_data(update=False,save_data=False)
    features.expand_datetime_features(df,df['date'])
    features.expand_features(df)
    feature_cols = df.columns
    # Remove, date, y_nb and y_sb because they are answers. Remove precipType,precipAccumulation, and cloudCover for lack of data
    trivial_columns = ['icon','summary','time']
    feature_cols = feature_cols.drop(trivial_columns)
    full_bike_data = features.clean_features(df[feature_cols], nan_perc_per_feature_thresh=500)
    io.save_file('feature_file',full_bike_data)

if __name__=="__main__":
    main()