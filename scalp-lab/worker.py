import argparse
import json
import os
import time

from runner import run_task


def save_progress(q, start):
    done = sum(1 for t in q if t["status"] == "done")
    err = sum(1 for t in q if t["status"] == "error")
    prog = {"total": len(q), "done": done, "errors": err,
            "elapsed_min": round((time.time() - start) / 60, 1),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs("results", exist_ok=True)
    with open("results/progress.json", "w") as f:
        json.dump(prog, f)
    hist = []
    try:
        with open("results/history.json") as f:
            hist = json.load(f)
    except Exception:
        hist = []
    if not hist or hist[-1].get("done") != done:
        hist.append({"updated": prog["updated"], "done": done,
                     "total": len(q), "errors": err})
        with open("results/history.json", "w") as f:
            json.dump(hist[-500:], f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=40)
    args = ap.parse_args()
    start = time.time()
    if not os.path.exists("results/queue.json"):
        from queue_builder import build
        os.makedirs("results", exist_ok=True)
        with open("results/queue.json", "w") as f:
            json.dump(build(), f)
    with open("results/queue.json") as f:
        q = json.load(f)
    done = 0
    for t in q:
        if t["status"] != "pending":
            continue
        if time.time() - start > args.budget * 60:
            break
        t["status"] = "running"
        try:
            res = run_task(t)
            out = os.path.join("results", t["kind"], t["id"] + ".json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(res, f)
            t["status"] = "done"
            t["result"] = out
            done += 1
            print("done", t["id"])
        except Exception as e:
            t["status"] = "error"
            t["error"] = str(e)[:300]
            print("error", t["id"], str(e)[:120])
        with open("results/queue.json", "w") as f:
            json.dump(q, f, ensure_ascii=False)
        save_progress(q, start)
    save_progress(q, start)
    print("processed", done)


if __name__ == "__main__":
    main()
