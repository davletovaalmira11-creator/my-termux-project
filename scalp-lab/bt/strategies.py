import numpy as np
import pandas as pd


def ema_cross(df, fast, slow):
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    return (f > s).astype(float) * 2 - 1


def bollinger_reversion(df, n, k):
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] < m - k * sd] = 1.0
    sig[df["close"] > m + k * sd] = -1.0
    sig[(df["close"] - m).abs() < 0.1 * sd] = 0.0
    return sig.ffill().fillna(0.0)


def donchian(df, n):
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    mid = (hi + lo) / 2
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] < mid] = 0.0
    sig[df["close"] > mid] = 0.0
    sig[df["close"] > hi] = 1.0
    sig[df["close"] < lo] = -1.0
    return sig.ffill().fillna(0.0)


def volume_surge(df, z, h):
    v = df["volume"]
    m = v.rolling(200).mean()
    s = v.rolling(200).std()
    ev = v > m + z * s
    dirn = np.sign(df["close"] - df["open"])
    sig = dirn.where(ev & (dirn != 0))
    return sig.ffill(limit=h).fillna(0.0)


def vwap_reversion(df, n, k):
    pv = df["quote_vol"].rolling(n).sum()
    v = df["volume"].rolling(n).sum()
    vwap = pv / v
    dev = df["close"] - vwap
    zscore = dev / dev.rolling(200).std()
    sig = pd.Series(np.nan, index=df.index)
    sig[zscore > k] = -1.0
    sig[zscore < -k] = 1.0
    sig[zscore.abs() < 0.1] = 0.0
    return sig.ffill().fillna(0.0)


def grid(**kv):
    res = [{}]
    for key in list(kv):
        new = []
        for d in res:
            for v in kv[key]:
                nd = dict(d)
                nd[key] = v
                new.append(nd)
        res = new
    return res


STRATEGIES = {
    "ema_cross": (ema_cross, grid(fast=[5, 9, 12], slow=[21, 30, 50])),
    "bollinger_reversion": (bollinger_reversion, grid(n=[20, 50], k=[2.0, 2.5, 3.0])),
    "donchian": (donchian, grid(n=[20, 40, 60])),
    "volume_surge": (volume_surge, grid(z=[2.5, 3.0, 4.0], h=[3, 5, 10])),
    "vwap_reversion": (vwap_reversion, grid(n=[30, 60], k=[1.5, 2.0, 2.5])),
}
