import argparse
import json
import os
import time

from runner import run_task

QUEUE_FILES = ["results/queue.json", "results/queue2.json", "results/queue3.json"]


def save_progress(all_tasks, start):
    done = sum(1 for t in all_tasks if t["status"] == "done")
    err = sum(1 for t in all_tasks if t["status"] == "error")
    prog = {"total": len(all_tasks), "done": done, "errors": err,
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
                     "total": len(all_tasks), "errors": err})
        with open("results/history.json", "w") as f:
            json.dump(hist[-500:], f)


def load_queues():
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(QUEUE_FILES[0]):
        from queue_builder import build
        with open(QUEUE_FILES[0], "w") as f:
            json.dump(build(), f)
    queues = []
    for p in QUEUE_FILES:
        if os.path.exists(p):
            with open(p) as f:
                queues.append((p, json.load(f)))
    return queues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=40)
    args = ap.parse_args()
    start = time.time()
    os.makedirs("reports", exist_ok=True)
    queues = load_queues()
    all_tasks = [t for _, q in queues for t in q]
    done = 0
    for path, q in queues:
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
            with open(path, "w") as f:
                json.dump(q, f, ensure_ascii=False)
            save_progress(all_tasks, start)
        if time.time() - start > args.budget * 60:
            break
    save_progress(all_tasks, start)
    print("processed", done)


if __name__ == "__main__":
    main()
