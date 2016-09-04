# Preprocessing tools
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

#Feature Selection tools
from sklearn.feature_selection import RFE,VarianceThreshold,GenericUnivariateSelect,SelectFromModel,chi2,f_classif
from sklearn.cross_validation import StratifiedKFold, cross_val_score

# Classification Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression, bayes, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, BaggingRegressor
