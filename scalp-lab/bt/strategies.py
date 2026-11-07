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


# ===== fee-aware стратегии: редкие сделки, длинное удержание =====


def streak_reversion(df, n_streak, h):
    # паттерн down3/up3: серия одноцветных свечей -> возврат к среднему
    up = (df["close"] - df["open"]) > 0
    dn = ~up
    up_run = up.groupby((~up).cumsum()).cumsum()
    dn_run = dn.groupby((~dn).cumsum()).cumsum()
    sig = pd.Series(np.nan, index=df.index)
    sig[dn_run >= n_streak] = 1.0
    sig[up_run >= n_streak] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def vol_drift(df, z, h):
    # паттерн vol_z>3: после всплеска объёма дрейф вверх до 60 баров
    v = df["volume"]
    m = v.rolling(200).mean()
    s = v.rolling(200).std()
    ev = v > m + z * s
    sig = pd.Series(np.nan, index=df.index)
    sig[ev] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def wick_reversion(df, w, h):
    # большие тени -> откат; вход лимиткой, maker fee
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    up_wick = df["high"] - body_hi
    lo_wick = body_lo - df["low"]
    sig = pd.Series(np.nan, index=df.index)
    sig[up_wick > w * rng] = -1.0
    sig[lo_wick > w * rng] = 1.0
    return sig.ffill(limit=h).fillna(0.0)


def donchian_trend(df, n):
    # тренд с большим окном: в разы меньше сделок -> комиссии меньше грузят
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    mid = (hi + lo) / 2
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] < mid] = 0.0
    sig[df["close"] > mid] = 0.0
    sig[df["close"] > hi] = 1.0
    sig[df["close"] < lo] = -1.0
    return sig.ffill().fillna(0.0)


def bb_maker(df, n, k):
    # bollinger с длинным окном: меньше сигналов, каждый крупнее комиссий;
    # исполнение лимитками у границ канала -> maker fee
    return bollinger_reversion(df, n, k)


# ===== комбинированные: сигнал + подтверждение объёмом (vol_z, FDR-выживший) =====

def _vol_z(df):
    v = df["volume"]
    m = v.rolling(200).mean()
    s = v.rolling(200).std().replace(0.0, np.nan)
    return (v - m) / s


def streak_vol(df, n_streak, vz, h):
    # серия одноцветных свечей + всплеск объёма -> реверс (объединение 2 FDR-паттернов)
    up = (df["close"] - df["open"]) > 0
    dn = ~up
    up_run = up.groupby((~up).cumsum()).cumsum()
    dn_run = dn.groupby((~dn).cumsum()).cumsum()
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(dn_run >= n_streak) & (z > vz)] = 1.0
    sig[(up_run >= n_streak) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def bb_vol(df, n, k, vz, h):
    # выход за границу Bollinger, подтверждённый объёмом: меньше ложных входов
    m = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[(df["close"] < m - k * sd) & (z > vz)] = 1.0
    sig[(df["close"] > m + k * sd) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


def wick_vol(df, w, vz, h):
    # большая тень + всплеск объёма -> откат
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    z = _vol_z(df)
    sig = pd.Series(np.nan, index=df.index)
    sig[((body_lo - df["low"]) > w * rng) & (z > vz)] = 1.0
    sig[((df["high"] - body_hi) > w * rng) & (z > vz)] = -1.0
    return sig.ffill(limit=h).fillna(0.0)


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
    # fee-aware волна
    "streak_reversion": (streak_reversion, grid(n_streak=[3, 4], h=[15, 30, 60])),
    "vol_drift": (vol_drift, grid(z=[3.0, 4.0], h=[30, 60, 120])),
    "wick_reversion": (wick_reversion, grid(w=[0.4, 0.5], h=[15, 30, 60])),
    "donchian_trend": (donchian_trend, grid(n=[96, 192, 384])),
    "bb_maker": (bb_maker, grid(n=[50, 100, 200], k=[2.0, 2.5])),
    # комбинированные: сигнал + объёмное подтверждение
    "streak_vol": (streak_vol, grid(n_streak=[3, 4], vz=[2.0, 3.0], h=[15, 30, 60])),
    "bb_vol": (bb_vol, grid(n=[50, 100], k=[2.0, 2.5], vz=[2.0, 3.0], h=[15, 30, 60])),
    "wick_vol": (wick_vol, grid(w=[0.4, 0.5], vz=[2.0, 3.0], h=[15, 30, 60])),
}

# режим исполнения: реверсивные входят лимитками (maker),
# трендовые - по рынку (taker)
MAKER_SET = {"streak_reversion", "wick_reversion", "bb_maker",
             "bollinger_reversion", "vwap_reversion",
             "streak_vol", "bb_vol", "wick_vol"}
