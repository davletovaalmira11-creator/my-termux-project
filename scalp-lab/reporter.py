import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

MIN_TRADES = 200
MIN_FOLDS = 8


def p_from_t(t):
    return math.erfc(abs(t) / math.sqrt(2.0))


def bh_keep(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    ok = np.where(ranked <= thresh)[0]
    if len(ok) == 0:
        return []
    kmax = int(ok.max())
    return [int(i) for i in order[:kmax + 1]]


def fmt(x):
    return ("%.4f" % x) if abs(x) < 100 else ("%.1f" % x)


def fee_mode(r):
    return r.get("fee_mode", "taker")


def maker_sharpe(r):
    m = r.get("oos_maker")
    return m["sharpe"] if isinstance(m, dict) else ""


def maker_pf(r):
    m = r.get("oos_maker")
    return m["pf"] if isinstance(m, dict) else ""


def main():
    os.makedirs("reports", exist_ok=True)
    bts = []
    for f in glob.glob("results/backtest/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if isinstance(r, dict) and "oos" in r:
            bts.append(r)
    scans = []
    for f in glob.glob("results/scan/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        for x in r.get("findings", []):
            y = dict(x)
            y["symbol"] = r.get("symbol")
            y["tf"] = r.get("tf")
            scans.append(y)

    lines = []
    lines.append("# Crypto Scalp Lab - отчет")
    lines.append("")
    lines.append("Все результаты - только out-of-sample (walk-forward), с учетом комиссий и проскальзывания.")
    lines.append("maker = вход лимитными ордерами (0.02%), taker = по рынку (0.05%).")
    lines.append("")

    lines.append("## Прогресс")
    try:
        prog = json.load(open("results/progress.json"))
        lines.append("- готово задач: %d из %d (ошибок: %d), обновлено %s" % (
            prog.get("done", 0), prog.get("total", 0), prog.get("errors", 0), prog.get("updated", "")))
    except Exception:
        lines.append("- пока нет данных")
    lines.append("")

    passed = [r for r in bts
              if r["trades"] >= MIN_TRADES
              and r["folds"] >= MIN_FOLDS
              and r["oos"]["pf"] > 1.0
              and r["oos"]["t"] > 1.0]
    passed.sort(key=lambda r: r["oos"]["sharpe"], reverse=True)

    lines.append("## Лидерборд стратегий (OOS)")
    lines.append("")
    lines.append("| # | стратегия | пара | TF | Sharpe | PF | WinRate | MaxDD | t | сделок | стаб. | режим | maker Sharpe | maker PF |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(passed[:30]):
        o = r["oos"]
        robust = "*" if (r["param_stability"] >= 0.5 and o["t"] > 2.0) else ""
        ms = maker_sharpe(r)
        mp = maker_pf(r)
        ms_s = ("%.2f" % ms) if ms != "" else "-"
        mp_s = ("%.2f" % mp) if mp != "" else "-"
        lines.append("| %d | %s%s | %s | %s | %.2f | %.2f | %.1f%% | %s | %.1f | %d | %.0f%% | %s | %s | %s |" % (
            i + 1, r["strategy"], robust, r["symbol"], r["tf"], o["sharpe"], o["pf"],
            o["winrate"] * 100, fmt(o["maxdd"]), o["t"], r["trades"], r["param_stability"] * 100,
            fee_mode(r), ms_s, mp_s))
    lines.append("")
    lines.append("Робастные отмечены звездочкой: стабильные параметры между фолдами + t>2.")
    lines.append("")

    lines.append("## Устойчивость по монетам (агрегация)")
    lines.append("")
    lines.append("Закономерность живет на многих парах; подгонка - обычно на одной.")
    lines.append("")
    lines.append("| стратегия | TF | пар | медиана Sharpe | медиана PF | пар с Sharpe>0 | режим |")
    lines.append("|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in passed:
        groups[(r["strategy"], r["tf"])].append(r)
    for (st, tf), xs in sorted(groups.items()):
        sh = [x["oos"]["sharpe"] for x in xs]
        pf = [x["oos"]["pf"] for x in xs]
        lines.append("| %s | %s | %d | %.2f | %.2f | %d | %s |" % (
            st, tf, len(xs), float(np.median(sh)), float(np.median(pf)),
            sum(1 for s in sh if s > 0), fee_mode(xs[0])))
    lines.append("")

    if scans:
        pvals = [p_from_t(x["t"]) for x in scans]
        keep = set(bh_keep(pvals))
        surv = [x for i, x in enumerate(scans) if i in keep]
        groups2 = defaultdict(list)
        for x in surv:
            groups2[(x["event"], x["horizon"])].append(x)
        lines.append("## Паттерны, выжившие поправку Бенджамини-Хохберга (FDR 5%%)")
        lines.append("")
        if groups2:
            lines.append("| событие | горизонт(баров) | пар | средн. форвард | медиана t |")
            lines.append("|---|---|---|---|---|")
            for (ev, hz), xs in sorted(groups2.items(), key=lambda kv: -abs(np.mean([a["mean_fwd"] for a in kv[1]]))):
                mf = np.mean([a["mean_fwd"] for a in xs])
                mt = np.median([a["t"] for a in xs])
                lines.append("| %s | %d | %d | %s | %.1f |" % (ev, hz, len(xs), fmt(mf), mt))
        else:
            lines.append("Пока ни один паттерн не прошел статзначимую проверку (это нормальный результат - зато без подгонок).")
        lines.append("")

    with open("reports/REPORT.md", "w") as f:
        f.write(chr(10).join(lines))

    with open("reports/leaderboard.csv", "w") as f:
        f.write("strategy,symbol,tf,oos_sharpe,oos_pf,oos_t,trades,param_stability,is_sharpe_avg,folds,fee_mode,maker_sharpe,maker_pf" + chr(10))
        for r in passed[:200]:
            o = r["oos"]
            f.write(",".join(str(x) for x in [r["strategy"], r["symbol"], r["tf"], round(o["sharpe"], 3), round(o["pf"], 3), round(o["t"], 2), r["trades"], round(r["param_stability"], 2), round(r["is_sharpe_avg"], 2), r["folds"], fee_mode(r), maker_sharpe(r), maker_pf(r)]) + chr(10))

    print("backtest results:", len(bts), "passed filters:", len(passed), "scan findings:", len(scans))


if __name__ == "__main__":
    main()
