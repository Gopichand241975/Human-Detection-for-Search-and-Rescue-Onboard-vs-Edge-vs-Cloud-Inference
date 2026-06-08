"""
latency_eval.py
───────────────
Merges the client-side network-delay CSV (preprocess_ms, network_delay)
with the server-side latency CSV (inference_latency) on
image_id and writes a combined report per run.

Output columns
──────────────
image_id, preprocess_ms, network_delay_ms, queue_wait_ms,
inference_latency_ms, total_latency_ms
"""

import pandas as pd
import glob
import os
import re


OUTPUT_DIR = "evaluation/results/latency"


# ---------- helpers ----------

def extract_run_id(filename):
    match = re.search(r"run_(\d{8}_\d{6})", filename)
    return match.group(1) if match else None


def normalize_image_id(x):
    return x.split("run_")[-1] if "run_" in x else x


# ---------- loaders ----------

def load_network(run_id):
    file = f"data/client_metrics/run_{run_id}_network-delay.csv"
    if not os.path.exists(file):
        return None

    df = pd.read_csv(file)

    if not {"image_id", "network_delay"}.issubset(df.columns):
        return None

    keep = ["image_id", "network_delay"]
    if "preprocess_ms" in df.columns:
        keep.append("preprocess_ms")

    df = df[keep].copy()
    df["image_id"] = df["image_id"].apply(normalize_image_id)

    df["network_delay"] = df["network_delay"] * 1000
    df = df.rename(columns={"network_delay": "network_delay_ms"})

    return df


def load_inference(run_id):
    file = f"data/metrics/run_{run_id}_latency.csv"
    if not os.path.exists(file):
        return None

    df = pd.read_csv(file)

    if not {"image_id", "latency"}.issubset(df.columns):
        return None

    keep = ["image_id", "latency"]
    if "queue_wait_ms" in df.columns:
        keep.append("queue_wait_ms")

    df = df[keep].copy()
    df["image_id"] = df["image_id"].apply(normalize_image_id)

    df["latency"] = df["latency"] * 1000
    df = df.rename(columns={"latency": "inference_latency_ms"})

    return df


# ---------- evaluate ----------

def evaluate_run(run_id):
    net = load_network(run_id)
    inf = load_inference(run_id)

    if net is None or inf is None:
        print(f"[SKIP] run_{run_id} missing required data")
        return

    merged = pd.merge(net, inf, on="image_id", how="inner")

    if merged.empty:
        print(f"[SKIP] run_{run_id} no matching image_ids")
        return

    # Build total_latency_ms: preprocess + network + queue_wait + inference
    total = merged["network_delay_ms"] + merged["inference_latency_ms"]

    if "preprocess_ms" in merged.columns:
        total = total + merged["preprocess_ms"]

    if "queue_wait_ms" in merged.columns:
        total = total + merged["queue_wait_ms"]

    merged["total_latency_ms"] = total

    # Column order (unchanged behavior)
    cols = ["image_id"]
    for optional in ("preprocess_ms", "network_delay_ms", "queue_wait_ms", "inference_latency_ms"):
        if optional in merged.columns:
            cols.append(optional)
    cols.append("total_latency_ms")

    merged = merged[cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_file = f"{OUTPUT_DIR}/run_{run_id}_latency.csv"
    merged.to_csv(out_file, index=False)

    print(f"[OK] Saved {out_file}  ({len(merged)} rows)")


# ---------- discovery ----------

def get_all_runs():
    # ✅ Primary path (original behavior)
    files = glob.glob("data/client_metrics/run_*_network-delay.csv")

    # ✅ Fallback (ONLY if nothing found — safe extension)
    if not files:
        files = glob.glob("**/data/client_metrics/run_*_network-delay.csv", recursive=True)

    runs = []
    for f in files:
        run_id = extract_run_id(os.path.basename(f))
        if run_id:
            runs.append(run_id)

    return sorted(set(runs))


# ---------- main ----------

def main():
    runs = get_all_runs()

    if not runs:
        print("[ERROR] No runs found")
        return

    for run_id in runs:
        evaluate_run(run_id)


if __name__ == "__main__":
    main()