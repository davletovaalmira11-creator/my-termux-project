import json

import config
from bt import data, engine, scan, strategies


def key_of(params):
    return json.dumps(params, sort_keys=True)


def run_task(t):
    tf = t["tf"]
    tf_min = int(tf[:-1])
    df = data.klines(t["symbol"], tf, t["days"])
    if len(df) < 5000:
        raise RuntimeError("not enough data")
    if t["kind"] == "backtest":
        fn, g = strategies.STRATEGIES[t["strategy"]]
        g = [p for p in g if p.get("fast", 1) < p.get("slow", 9999)]
        sigs = {}
        for p in g:
            try:
                sigs[key_of(p)] = fn(df, **p)
            except Exception:
                pass
        train_bars = int(t["train_days"] * 24 * 60 / tf_min)
        test_bars = int(t["test_days"] * 24 * 60 / tf_min)
        res = engine.walk_forward(df, sigs, train_bars, test_bars,
                                 config.FEE, config.SLIP, tf_min)
        res.update({"kind": "backtest", "symbol": t["symbol"], "tf": tf,
                    "strategy": t["strategy"], "days": int(t["days"]),
                    "n_bars": int(len(df))})
        return res
    res = scan.run_scan(df, t["symbol"], tf)
    return res
