import argparse

import utils.io as io
import utils.data as data
import utils.features as features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-data", action="store_true")
    parser.add_argument("--pull-weather", action="store_true")
    parser.add_argument("--expand-features", type=bool, default=True)
    parser.add_argument("-v","--verbose", action="store_true")
    args = parser.parse_args()
    update = args.pull_data
    expand_feautures = args.pull_weather
    verbose = args.pull_data
    df = data.load_bike_data(update=False,save_data=False)


    features.expand_datetime_features(df,df['date'])
    features.convert_weather_summary(df)
    features.highlight_peaks(df)
    feature_cols = df.columns
    trivial_columns = ['icon','summary','time']
    feature_cols = feature_cols.drop(trivial_columns)
    full_bike_data = features.clean_features(df[feature_cols], nan_perc_per_feature_thresh=500)
    io.save_file('feature_file',full_bike_data)

if __name__=="__main__":
    main()