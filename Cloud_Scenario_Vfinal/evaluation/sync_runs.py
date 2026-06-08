import os
import json
from datetime import datetime

PRED_DIR = "data/predictions"
LATENCY_DIR = "data/metrics"
RUN_FILE = "data/runs.json"


def extract_created(run_id):
    try:
        ts = run_id.replace("run_", "")
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


def scan_runs():
    runs = []

    if not os.path.exists(PRED_DIR):
        return runs

    for name in os.listdir(PRED_DIR):
        path = os.path.join(PRED_DIR, name)

        if not os.path.isdir(path) or not name.startswith("run_"):
            continue

        run_id = name

        # predictions check
        has_predictions = any(
            f.endswith(".json") for f in os.listdir(path)
        )

        # latency check
        latency_file = os.path.join(LATENCY_DIR, f"{run_id}_latency.csv")
        has_latency = os.path.exists(latency_file)

        runs.append({
            "run_id": run_id,
            "created": extract_created(run_id),
            "has_predictions": has_predictions,
            "has_latency": has_latency
        })

    return sorted(runs, key=lambda x: x["run_id"])


def rebuild_runs_json():
    runs = scan_runs()

    data = {"runs": runs}

    os.makedirs("data", exist_ok=True)

    tmp = RUN_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)

    os.replace(tmp, RUN_FILE)

    print(f"[SYNC] runs.json updated ({len(runs)} runs)")


if __name__ == "__main__":
    rebuild_runs_json()