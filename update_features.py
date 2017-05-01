import argparse
import utils.data
import utils.features
import utils.io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-new-data", action="store_true")
    parser.add_argument("--expand-features", type=bool, default=True)
    parser.add_argument("-v","--verbose", action="store_true")
    args = parser.parse_args()
    update = args.pull_new_data
    expand_features = args.expand_features
    verbose = args.verbose
    bike_data = utils.data.load_bike_data(update=update, save_data=update, verbose=verbose)

    if expand_features:
        utils.features.expand_datetime_features(bike_data, bike_data['date'])
        utils.features.convert_weather_summary(bike_data)
        utils.features.highlight_peaks(bike_data)
        feature_cols = bike_data.columns
        trivial_columns = ['icon', 'summary', 'time']
        feature_cols = feature_cols.drop(trivial_columns)
        full_bike_data = utils.features.clean_features(bike_data[feature_cols], nan_perc_per_feature_thresh=500)
        utils.io.save_file('feature_file',full_bike_data)


if __name__=="__main__":
    main()