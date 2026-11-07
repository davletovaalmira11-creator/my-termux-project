"""Paper trading: реальные сигналы топ-стратегий на живых данных Binance.

Каждый запуск (воркер каждые 4 часа):
- берет топ-стратегии из результатов бэктестов (прошедшие фильтры);
- на свежих данных генерирует сигнал последнего закрытого бара;
- открывает/закрывает виртуальные позиции, капитал $1000, сложный процент;
- пишет results/paper.json для панели.
"""
import glob
import json
import os
import time

import config
from bt import data, strategies
try:
    from bt import live
except Exception:
    live = None

STATE_PATH = "results/paper.json"
CAPITAL_START = 1000.0
MAX_TRACK = 5
MIN_STABILITY = 0.3
LOOKBACK_DAYS = {"1m": 3, "5m": 10, "15m": 20, "1h": 40}
COST = {"taker": 2 * (config.FEE + config.SLIP),
        "maker": 2 * (config.FEE_MAKER + config.SLIP_MAKER)}


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                st = json.load(f)
            if "positions" in st and "closed" in st and "equity" in st:
                return st
        except Exception:
            pass
    return {"capital_start": CAPITAL_START, "equity": CAPITAL_START,
            "positions": {}, "closed": [], "updated": "", "runs": 0}


def save_state(st):
    os.makedirs("results", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def top_strategies():
    best = {}
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(r, dict) and "oos" in r):
            continue
        k = (r.get("strategy"), r.get("symbol"), r.get("tf"))
        try:
            mt = os.path.getmtime(f)
        except Exception:
            mt = 0
        if k not in best or mt >= best[k][0]:
            best[k] = (mt, r)
    bts = [v[1] for v in best.values()]
    for r in bts:
        o = r["oos"]
        if r.get("trades", 0) < 200 or r.get("folds", 0) < 8:
            continue
        if o.get("pf", 0) <= 1.0 or o.get("t", 0) <= 1.0:
            continue
        if r.get("param_stability", 0) < MIN_STABILITY:
            continue
        bts.append(r)
    bts.sort(key=lambda r: r["oos"]["sharpe"], reverse=True)
    out = []
    seen = set()
    for r in bts:
        key = (r["strategy"], r["symbol"], r["tf"])
        if key in seen:
            continue
        seen.add(key)
        try:
            params = json.loads(r.get("top_param", "{}"))
        except Exception:
            continue
        if not params:
            continue
        fee_mode = "maker" if r["strategy"] in strategies.MAKER_SET else "taker"
        out.append({"key": "-".join(key), "strategy": r["strategy"],
                    "symbol": r["symbol"], "tf": r["tf"],
                    "params": params, "fee_mode": fee_mode,
                    "oos_sharpe": round(r["oos"]["sharpe"], 2)})
        if len(out) >= MAX_TRACK:
            break
    return out


def close_pos(st, key, px, ts, reason):
    pos = st["positions"].pop(key)
    d = pos["signal"]
    pnl = d * (px / pos["entry"] - 1.0) - COST[pos["fee_mode"]]
    st["equity"] = st["equity"] * (1.0 + pnl)
    st["closed"].append({
        "key": key, "strategy": pos["strategy"], "symbol": pos["symbol"],
        "tf": pos["tf"], "side": "long" if d > 0 else "short",
        "entry": pos["entry"], "exit": px,
        "pnl_pct": round(pnl * 100, 4),
        "opened": pos["opened"], "closed_ts": ts,
        "fee_mode": pos["fee_mode"], "reason": reason})
    st["closed"] = st["closed"][-300:]


def get_klines(symbol, tf):
    """Живые свечи (лаг ~1 минута); при сбое - дневные архивы (лаг ~1 день)."""
    if live is not None:
        try:
            return live.klines_live(symbol, tf, LOOKBACK_DAYS.get(tf, 10))
        except Exception as e:
            print("live data failed, fallback to archives:", str(e)[:80], flush=True)
    return data.klines(symbol, tf, LOOKBACK_DAYS.get(tf, 10))


def main():
    st = load_state()
    tops = top_strategies()
    top_keys = set(t["key"] for t in tops)
    # закрыть позиции стратегий, выпавших из топа
    for key in list(st["positions"].keys()):
        if key in top_keys:
            continue
        t = st["positions"][key]
        try:
            df = get_klines(t["symbol"], t["tf"])
            close_pos(st, key, float(df["close"].iloc[-1]),
                      str(df["dt"].iloc[-1]), "rotated out")
        except Exception:
            pass
    for t in tops:
        try:
            df = get_klines(t["symbol"], t["tf"])
            if t["strategy"] in strategies.DERIV_SET:
                df = data.add_deriv(df, t["symbol"], LOOKBACK_DAYS.get(t["tf"], 10))
            fn = strategies.STRATEGIES[t["strategy"]][0]
            sig = fn(df, **t["params"])
            cur = float(sig.iloc[-1])
            t["signal"] = cur
        except Exception as e:
            print("paper skip", t["key"], str(e)[:80], flush=True)
            continue
        px = float(df["close"].iloc[-1])
        ts = str(df["dt"].iloc[-1])
        pos = st["positions"].get(t["key"])
        if cur != 0.0:
            if pos is None or pos["signal"] != cur:
                if pos is not None:
                    close_pos(st, t["key"], px, ts, "signal flip")
                st["positions"][t["key"]] = {
                    "strategy": t["strategy"], "symbol": t["symbol"],
                    "tf": t["tf"], "signal": cur, "entry": px,
                    "opened": ts, "fee_mode": t["fee_mode"],
                    "oos_sharpe": t["oos_sharpe"], "last_ts": ts,
                    "last_px": px}
            else:
                pos["last_ts"] = ts
                pos["last_px"] = px
        elif pos is not None:
            close_pos(st, t["key"], px, ts, "flat signal")
    st["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    st["runs"] = int(st.get("runs", 0)) + 1
    st["tops"] = tops
    trades = st["closed"]
    wins = sum(1 for x in trades if x["pnl_pct"] > 0)
    st["stats"] = {
        "n_trades": len(trades),
        "wins": wins,
        "winrate": round(100.0 * wins / len(trades), 1) if trades else 0.0,
        "equity": round(st["equity"], 2),
        "pnl_pct": round((st["equity"] / st["capital_start"] - 1.0) * 100, 2),
        "open": len(st["positions"])}
    save_state(st)
    print("paper: equity %.2f, trades %d, open %d" % (
        st["equity"], len(st["closed"]), len(st["positions"])), flush=True)


if __name__ == "__main__":
    main()
