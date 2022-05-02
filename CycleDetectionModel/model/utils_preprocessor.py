import time
import numpy as np
import pandas as pd

# # signal processing
w = pd.Series([1, 2, 4, 2, 1])
pct = [.999, .995, .99, .98, .95]

def rolling(timeseries, freq, w):
    """
    inputs
      timeseries
        a pandas Series, must have DatetimeIndex
      freq
        a frequency string to be used as downsampling rate (default 1)
      w
        a list of weights that sum up to 1 (automatically rescaled if not)

    output
      a pandas Series with DatetimeIndex (difference)
      a pandas Series with DatetimeIndex (subtrahend)
    """
    def mwa(arr, w=pd.Series(w)):
        """ get midpoint weighted average """
        if len(arr) != len(w):
            return arr.mean()
        arr_sum = arr.reset_index(drop=True).multiply(w).sum()
        w_sum = w[arr.notna().values].sum()
        return arr_sum / w_sum if w_sum else np.nan

    series = (resampler := timeseries.resample(freq)).apply(np.mean)
    freq = resampler.freq
    series_out = []
    for t in range(tlen := len(w)):
        srs = (resamp := series.resample((freq * tlen), offset=(freq * t))).apply(mwa)
        srs.index += (resamp.freq - freq).delta / 2
        series_out.append(srs)
    series_new = pd.concat(series_out)[series.index].sort_index()
    series_sub = series - series_new
    return series_new, series_sub


def downsampling(timeseries, freq, applyfunc=np.median):
    """
    inputs
      timeseries
        a pandas Series, must have DatetimeIndex
      freq
        a frequency string to be used as downsampling rate
      applyfunc
        a function to use for aggregating the data
        see pandas.core.resample.Resampler.apply

    output
      a pandas Series with DatetimeIndex
    """
    series_new = (resampler := timeseries.resample(freq)).apply(applyfunc)
    series_new.index += resampler.freq.delta / 2
    return series_new


def trimming(timeseries, baseline, pct, step=.005):
    """
    inputs
      timeseries
        a pandas Series, must have DatetimeIndex
      baseline
        a pandas Series, must have DatetimeIndex
        should have lower sampling rate than the original timeseries
      pct
        a percentile, or a list of percentiles
        If pct is a percentile, a list of values are generated
        within the half-open interval [1 - pct, 1)
      step
        The distance between two adjacent values.
        The default step size is 1. If pct is a list, step is ignored.
    output
      a pandas Series with DatetimeIndex (difference)
      a pandas Series with DatetimeIndex (subtrahend)
    """
    if not hasattr(pct, '__iter__'):
        pct = (pct := np.arange(1 - pct, 1, step)[::-1])[pct < 1]
    idx_union = (idx_og := timeseries.index).union(baseline.index)
    base_aligned = baseline.reindex(idx_union).interpolate().loc[idx_og].bfill()
    series_sub = timeseries - base_aligned
    outlier_cutoffs = series_sub.abs().quantile(pct)
    outer_ls = list()
    for orient in 1, -1:
        inner_ls, resid = list(), series_sub.mul(orient)
        for cutoff in outlier_cutoffs:
            idx = resid > cutoff
            resid[idx] -= (adj := (resid[idx] - cutoff) / 2)
            inner_ls.append(adj)
        outer_ls.append(pd.concat(inner_ls, axis=1).sum(axis=1).mul(orient))
    adj = pd.concat(outer_ls, axis=1).sum(axis=1)
    series_new = timeseries.sub(adj, fill_value=0).rename(timeseries.name)
    return series_new, series_sub


def signal_preprocessing(data, w=w, pct=pct):
    data_ = data[['timestamp', 'value']].set_index('timestamp').value
    s01, s01_reduced = rolling(data_, freq='1s', w=w)
    s10 = downsampling(s01, freq='10s')
    s30 = downsampling(s01, freq='30s')
    s10_clean, s10_spike = trimming(s10, baseline=s30, pct=pct)
    data = s10_clean.reset_index()
    return data