import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error,explained_variance_score

import utils.gen as gen

def display_results(regressors, X, y, ax=None, **line_kwargs):
    if ax is None: f,ax = plt.subplots()
    ax.plot(y,linewidth=2,c='k')
    for regressor_type,clf in regressors.iteritems():
        pred_y = clf.predict(X)
        ax.plot(pred_y,label=regressor_type,**line_kwargs)
    ax.legend()
    ax.set_title('Results')


# TODO: Make dataframe of coefficient data
def display_coefs(regressors, feature_cols, ax=None, **line_kwargs):
    feature_coefs = pd.DataFrame(index=feature_cols,columns=regressors.keys())

    if ax is None: f,ax = plt.subplots()
    for regressor_type,clf in regressors.iteritems():
        try:
            feature_coefs.loc[feature_cols,regressor_type] = clf.coef_
        except:
            feature_coefs.loc[feature_cols,regressor_type] = clf.feature_importances_
    feature_coefs.plot(ax=ax,**line_kwargs)
    ax.legend()
    ax.set_title('Feature Coefficient Distribution')


def basic_results(X,y,regressors,n_folds=3,analysis_func_type='r2_score',random_state=1223,verbose=False):
    kf = KFold(y,n_splits=gen.clamp(n_folds,2,10,int),random_state=random_state)
    if analysis_func_type == 'r2_score':
        analysis_func = r2_score
    elif analysis_func_type == 'mean2_squared_error':
        analysis_func = mean_squared_error
    elif analysis_func_type == 'explained_variance':
        analysis_func = explained_variance_score
    else:
        analysis_func = r2_score
    cols = ["Fold "+str(ifold+1) for ifold in range(kf.n_folds)]+['MEAN','BEST','WORST']
    results = pd.DataFrame(index=regressors.keys(),columns=cols)
    for regressor_type,clf in regressors.iteritems():
        for i,(itrain,itest) in enumerate(kf):
            predy = clf.fit(X[itrain],y[itrain]).predict(X[itest])
            results.loc[regressor_type,cols[i]] = analysis_func(y[itest],predy)
        results.loc[regressor_type,results.columns[:kf.n_folds]] = (clf,X,y)
        results.loc[regressor_type,'MEAN'] = results.loc[regressor_type,results.columns[:kf.n_folds]].mean()
        results.loc[regressor_type,'BEST'] = results.loc[regressor_type,results.columns[:kf.n_folds]].max()
        results.loc[regressor_type,'WORST'] = results.loc[regressor_type,results.columns[:kf.n_folds]].min()
    if verbose: print results
    return results


def full_results(X,y,regressors,n_folds=4,random_state=1234,verbose=False):
    kf = KFold(len(y),n_folds=n_folds,random_state=random_state)
    best_regressors = {k:None for k in regressors}
    minor_index= ["Fold-"+str(i+1) for i in range(kf.n_folds)]+['AVG']
    index = pd.MultiIndex.from_product([regressors.keys(),minor_index])
    res = pd.DataFrame(index=index,columns=['r2_score','mean2_squared_error','explained_variance'])
    for regressor_type,clf in regressors.iteritems():
        if verbose: print "Scoring {0}...".format(regressor_type)
        total_time = 0
        for i,(itrain,itest) in enumerate(kf):
            if verbose: print "Fold #",i,":\n\tTraining..."
            t_start = time.clock()
            predy = clf.fit(X[itrain],y[itrain]).predict(X[itest])
            t_end = time.clock()
            if verbose: print "\t...Finished, {0:.2f} seconds".format(t_end-t_start)
            total_time+=(t_end-t_start)
            res.loc[(regressor_type,minor_index[i]),:] = r2_score(y_true=y[itest],y_pred=predy),\
                                                          mean_squared_error(y_true=y[itest],y_pred=predy),\
                                                          explained_variance_score(y_true=y[itest],y_pred=predy)
            if res.loc[(regressor_type,minor_index[i]),'r2_score'] == res.loc[regressor_type]['r2_score'].max():
                best_regressors[regressor_type] = clf
        res.loc[(regressor_type,minor_index[-1]),:] = res.loc[regressor_type,:].mean(0)
        if verbose:
            print "\t Average Train Time: {0:.3f}".format(total_time/float(kf.n_folds))
            print "Results for {0}:".format(regressor_type)
            print res.loc[regressor_type,:]
    return res,best_regressors

