import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge,Ridge,Lasso, ElasticNet

from utils.data import load_bike_data
from utils.features import highlight_peaks,clean_features

# Evaluation Tools

rseedval = 1234

regressors = {
              # 'SGD_OLS_L2':SGDRegressor(loss='squared_loss',penalty='l2',l1_ratio=0.15,alpha=1e-6,random_state=rseedval),
              # 'SGDElasticNet_0.15':SGDRegressor(loss='huber',penalty='elasticnet',l1_ratio=0.15,random_state=rseedval),
              # 'SGDElasticNet_0.85':SGDRegressor(loss='huber',penalty='elasticnet',l1_ratio=0.85,random_state=rseedval),
              # 'SGD_eps_ins':SGDRegressor(loss='epsilon_insensitive',penalty='elasticnet',l1_ratio=0.15,random_state=rseedval), #SVR with elastic net regression
              'OLS_L2_Ridge':Ridge(alpha=1.0),
              'BayesianRidge':BayesianRidge(n_iter=500,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6),
              'Lasso':Lasso(alpha=1.0,random_state=rseedval),
              # 'LassoLars':LassoLars(alpha=1.0),
              'ElasticNet':ElasticNet(l1_ratio=0.15,random_state=rseedval),
              # 'SVR':SVR(kernel='rbf',C=1.0),
              'RandomForestRegressor':RandomForestRegressor(n_estimators=50,max_depth=6),
              # 'AdaBoostRegressor':AdaBoostRegressor(n_estimators=50),
              'ExtraTreesRegressor':ExtraTreesRegressor(n_estimators=50,max_depth=6)
              # 'GradientBoostingRegressor':GradientBoostingRegressor(n_estimators=50,max_depth=6)
              }

bike_data = load_bike_data()
highlight_peaks(bike_data)
y_nb = bike_data['fremont_bridge_nb']
y_sb = bike_data['fremont_bridge_sb']
y = bike_data['y']
feature_cols = bike_data.columns

# Remove, date, y_nb and y_sb because they are answers. Remove precipType,precipAccumulation, and cloudCover for lack of data
trivial_columns = ['date',
                   'icon',
                   'summary',
                   'time',
                   'fremont_bridge_nb',
                   'fremont_bridge_sb',
                   'y']
feature_cols = feature_cols.drop(trivial_columns)
bike_data,y = clean_features(bike_data[feature_cols],labels=y,nan_perc_per_feature_thresh=500)
feature_cols = bike_data.columns.tolist()
is_weekend = bike_data['is_weekend']

# Split weekday, weekend data
X1,y1 = bike_data.loc[~is_weekend,:],y[~is_weekend]
X2,y2 = bike_data.loc[is_weekend,:],y[is_weekend]

weekday_regressors = regressors
weekend_regressors = regressors

weekday_feature_cols = [col for col in feature_cols if 'weekend' not in col]
weekend_feature_cols = [col for col in feature_cols if 'weekday' not in col]

# Weekday Results ---------------
print "Weekday data"
# currX = X1[weekday_feature_cols].values
# curry = y1.values
# results,best_weekday_regressors = full_results(currX,curry,weekday_regressors,n_folds=3,verbose=True)
# print results
# f,ax1 = plt.subplots()
# display_results(best_weekday_regressors, currX, curry,ax=ax1, linestyle='--')
# f,ax2 = plt.subplots()
# display_coefs(best_weekday_regressors,weekday_feature_cols,marker='o',linestyle='--',ax=ax2)
# plt.show()

# Weekend Results ---------------
print "\n\n\nWeekend data"

# currX = X2[weekend_feature_cols].values
# curry = y2.values
# results,best_weekend_regressors = full_results(currX,curry,weekend_regressors,n_folds=3,verbose=True)
# print results
# f,ax1 = plt.subplots()
# display_results(best_weekend_regressors, currX, curry, ax=ax1, linestyle='--')
# f,ax2 = plt.subplots()
# display_coefs(best_weekend_regressors,weekend_feature_cols,marker='o', ax=ax2,linestyle='--')
# plt.show()