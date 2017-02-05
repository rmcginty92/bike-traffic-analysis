import sys
import numpy as np
import pandas as pd


def _peakdet(v, delta, x=None):
        """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html

        Returns two arrays

        function [maxtab, mintab]=peakdet(v, delta, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %
        %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        %        in MAXTAB and MINTAB are replaced with the corresponding
        %        X-values.
        %
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.

        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.

        """
        maxtab = []
        mintab = []

        if x is None:
            x = np.arange(len(v))

        v = np.asarray(v)

        if len(v) != len(x):
            sys.exit('Input vectors v and x must have same length')

        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')

        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        lookformax = True
        for i in np.arange(len(v)):
            this = v[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]

            if lookformax:
                if this < mx-delta:
                    maxtab.append((mxpos, mx))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    mintab.append((mnpos, mn))
                    mx = this
                    mxpos = x[i]
                    lookformax = True
        return np.array(maxtab), np.array(mintab)




def pandas_wind(data, N=None, fs=22050, tm_window=0.1, apply_func=np.sum,center=True,win_type=None,fill_nans=True,
                return_nans=False,applied_axis=0, **win_kwargs):
    orig_pd_obj = True
    if data.ndim == 2:
        if not isinstance(data,pd.core.frame.DataFrame):
            data = pd.DataFrame(data)
            orig_pd_obj = False
    elif data.ndim ==1:
        if not isinstance(data,pd.core.series.Series):
            data = pd.Series(data)
            orig_pd_obj = False
        applied_axis = 0
    else: return data
    if N is None:
        N = int(fs*tm_window)
        if N > len(data): return data
    N = int(N)
    def _aggregate(data,N,apply_func,win_type,center=center,applied_axis=0,special_kwargs={},**func_kwargs):
        try:
            if isinstance(apply_func,str):
                if apply_func == 'rms':windowed_data = (data**2).rolling(window=N,center=center,axis=applied_axis,win_type=win_type)
                else:windowed_data = data.rolling(window=N,center=center,axis=applied_axis,win_type=win_type)
                if apply_func == 'mean':aggregated_data = windowed_data.mean(**special_kwargs)
                elif apply_func == 'median':aggregated_data = windowed_data.median(**special_kwargs)
                elif apply_func == 'sum':aggregated_data = windowed_data.sum(**special_kwargs)
                elif apply_func == 'rms':aggregated_data = np.sqrt(windowed_data.mean(**special_kwargs).sqrt())
                elif apply_func == 'max':aggregated_data = windowed_data.max(**special_kwargs)
                elif apply_func == 'min':aggregated_data = windowed_data.min(**special_kwargs)
                elif apply_func == 'std':aggregated_data = windowed_data.std(**special_kwargs)
                elif apply_func == 'var':aggregated_data = windowed_data.var(**special_kwargs)
                elif apply_func == 'skew':aggregated_data = windowed_data.skew(**special_kwargs)
                elif apply_func == 'kurt':aggregated_data = windowed_data.kurt(**special_kwargs)
                elif apply_func == 'quantile':aggregated_data = windowed_data.quantile(func_kwargs.get('quantile',0.5),**special_kwargs)
                else: aggregated_data = windowed_data.mean(**special_kwargs)
            else:
                windowed_data = data.rolling(window=N,center=center,axis=applied_axis,win_type=win_type)
                aggregated_data = windowed_data.apply(apply_func,**special_kwargs)
        except:
            windowed_data = data.rolling(window=N,center=center,axis=applied_axis,win_type=win_type)
            aggregated_data = windowed_data.mean(**special_kwargs)
        return aggregated_data
    win_types = ['boxcar','triang','blackman','hamming','bartlett','parzen','bohman','blackmanharris',
                 'nuttall','barthann','kaiser','gaussian','general_gaussian','slepian']
    if win_type is not None and win_type not in win_types: win_type = None
    if win_type == 'kaiser':
        special_kwargs = {'beta':win_kwargs.get('beta',1)}
    elif win_type == 'gaussian':
        special_kwargs = {'std':win_kwargs.get('std',1)}
    elif win_type == 'general_gaussian':
        special_kwargs = {'power':win_kwargs.get('power',1),
                          'width':win_kwargs.get('width',1)}
    elif win_type == 'slepian':
        special_kwargs = {'width':win_kwargs.get('width',1)}
    else:
        special_kwargs = {}
    res = _aggregate(data,N,apply_func=apply_func,win_type=win_type,center=center,applied_axis=applied_axis,special_kwargs=special_kwargs,**win_kwargs)
    if fill_nans:
        res.fillna(inplace=True,method='ffill')
        res.fillna(inplace=True,method='bfill')
    if return_nans: index = res.index
    else: index = res.notnull()
    if orig_pd_obj: return res[index]
    else: return res[index].values