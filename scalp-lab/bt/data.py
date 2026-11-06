import os
import zipfile

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_vol", "trades", "taker_base", "taker_quote", "ignore"]
KEEP = ["dt", "open", "high", "low", "close", "volume", "quote_vol",
        "trades", "taker_base", "taker_quote"]

_MEMO = {}


def _fetch(url, path):
    r = requests.get(url, timeout=180)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return True


def _read_zip(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, names=COLS)
    # newer Binance archives include a header row - drop non-numeric rows
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    return df


def _klines_raw(symbol, tf, days, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)
    frames = []
    for m in pd.period_range(start, end, freq="M"):
        fn = symbol + "-" + tf + "-" + str(m) + ".zip"
        p = os.path.join(cache_dir, fn)
        url = BASE + "/monthly/klines/" + symbol + "/" + tf + "/" + fn
        if not os.path.exists(p):
            if not _fetch(url, p):
                continue
        frames.append(_read_zip(p))
    cur_start = end.to_period("M").start_time.tz_localize("UTC")
    for d in pd.date_range(cur_start, end, freq="D"):
        fn = symbol + "-" + tf + "-" + d.strftime("%Y-%m-%d") + ".zip"
        p = os.path.join(cache_dir, fn)
        url = BASE + "/daily/klines/" + symbol + "/" + tf + "/" + fn
        if not os.path.exists(p):
            if not _fetch(url, p):
                continue
        frames.append(_read_zip(p))
    if not frames:
        raise RuntimeError("no data for " + symbol + " " + tf)
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df = df[df["dt"] >= start].reset_index(drop=True)
    df = df[KEEP]
    for c in KEEP[1:]:
        df[c] = df[c].astype(float)
    return df


def klines(symbol, tf, days, cache_dir=".cache"):
    key = (symbol, tf, days)
    if key in _MEMO:
        return _MEMO[key]
    try:
        df = _klines_raw(symbol, tf, days, cache_dir)
    except RuntimeError:
        # some futures symbols are quoted as 1000<SYM>USDT (e.g. PEPE)
        if not symbol.endswith("USDT") or symbol.startswith("1000"):
            raise
        alt = "1000" + symbol
        df = _klines_raw(alt, tf, days, cache_dir)
    _MEMO[key] = df
    return df
