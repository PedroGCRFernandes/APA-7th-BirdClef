"""
experiment_log.py

Append-only experiment logger. Every experiment is a new line in experiments.jsonl
— old entries are never overwritten or deleted.

Each entry tracks:
- What the LLM proposed and generated
- Whether it succeeded or crashed (and how many times it crashed before a result)
- The performance metrics
- A human-readable label and notes for easy comparison across experiments
- A run_id grouping all experiments from the same agent session
"""

import json
import os
from datetime import datetime

LOG_PATH = "experiments.jsonl"


# ── Run ID ────────────────────────────────────────────────────────────────────
# Generated once per agent session. Groups all experiments from the same run.

def new_run_id():
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


# ── Write ──────────────────────────────────────────────────────────────────────

def add_experiment(
    run_id,
    iteration,
    label,               # short human-readable name  e.g. "baseline_cnn"
    architecture,        # plain-English description from the LLM
    code,                # the generated Python code
    status,              # "success" or "crashed"
    crash_count,         # how many times code crashed before this result
    error,               # error message if status == "crashed", else None
    val_auc,             # primary metric (None if crashed)
    val_loss,            # (None if crashed)
    epochs_trained,      # how many epochs actually ran
    training_time_sec,   # wall-clock seconds
    llm_analysis,        # LLM's own reflection on the result
    epoch_history=None,  # dict with keys loss/val_loss/auc/val_auc (lists per epoch)
    notes=""             # any free-text notes you want to attach
):
    entry = {
        "timestamp"        : datetime.now().isoformat(),
        "run_id"           : run_id,
        "iteration"        : iteration,
        "label"            : label,
        "notes"            : notes,
        "architecture"     : architecture,
        "status"           : status,
        "crash_count"      : crash_count,
        "error"            : error,
        "val_auc"          : val_auc,
        "val_loss"         : val_loss,
        "epochs_trained"   : epochs_trained,
        "training_time_sec": training_time_sec,
        "llm_analysis"     : llm_analysis,
        "epoch_history"    : epoch_history or {},
        "code"             : code,
    }

    # append one line — never overwrites existing entries
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    _print_summary(entry)
    return entry


# ── Read ───────────────────────────────────────────────────────────────────────

def load_all():
    """Return every experiment ever logged as a list of dicts."""
    if not os.path.exists(LOG_PATH):
        return []
    entries = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_run(run_id):
    """Return only the experiments from a specific agent session."""
    return [e for e in load_all() if e["run_id"] == run_id]


def load_by_label(label):
    """Return all experiments with a specific label across all runs."""
    return [e for e in load_all() if e["label"] == label]


def load_successful():
    """Return only experiments that completed without crashing."""
    return [e for e in load_all() if e["status"] == "success"]


# ── Labelling helper ───────────────────────────────────────────────────────────

def add_note(label, note):
    """
    Attach a note to the most recent experiment with a given label.
    Rewrites the file — use sparingly, only for human annotation.
    """
    entries = load_all()
    updated = False
    for entry in reversed(entries):
        if entry["label"] == label:
            entry["notes"] = note
            updated = True
            break
    if updated:
        _rewrite(entries)
        print(f"Note added to label '{label}'.")
    else:
        print(f"No experiment found with label '{label}'.")


# ── Summary helpers ────────────────────────────────────────────────────────────

def print_summary(run_id=None):
    """Print a readable table of all experiments (or one run)."""
    entries = load_run(run_id) if run_id else load_all()
    if not entries:
        print("No experiments logged yet.")
        return

    print(f"\n{'#':<5} {'Label':<25} {'Status':<10} {'Crashes':<9} {'AUC':<8} {'Time(s)':<10} {'Run ID'}")
    print("-" * 85)
    for e in entries:
        auc  = f"{e['val_auc']:.4f}" if e["val_auc"] is not None else "—"
        time = f"{e['training_time_sec']}" if e["training_time_sec"] is not None else "—"
        print(
            f"{e['iteration']:<5} "
            f"{e['label']:<25} "
            f"{e['status']:<10} "
            f"{e['crash_count']:<9} "
            f"{auc:<8} "
            f"{time:<10} "
            f"{e['run_id']}"
        )
    print()

    successful = [e for e in entries if e["val_auc"] is not None]
    if successful:
        best = max(successful, key=lambda e: e["val_auc"])
        print(f"Best so far → label='{best['label']}' | val_auc={best['val_auc']:.4f} | iteration={best['iteration']}")
    print()


def crash_report():
    """Show crash statistics across all experiments."""
    entries = load_all()
    if not entries:
        print("No experiments logged yet.")
        return

    total        = len(entries)
    total_crashes = sum(e["crash_count"] for e in entries)
    hard_crashes  = [e for e in entries if e["status"] == "crashed"]

    print(f"\nTotal experiments    : {total}")
    print(f"Total crash events   : {total_crashes}")
    print(f"Hard failures        : {len(hard_crashes)} (never recovered)")
    if hard_crashes:
        print("\nHard failure details:")
        for e in hard_crashes:
            print(f"  [{e['label']}] iteration={e['iteration']} → {e['error']}")
    print()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _print_summary(entry):
    auc = f"{entry['val_auc']:.4f}" if entry["val_auc"] is not None else "—"
    print(
        f"[{entry['run_id']} | iter {entry['iteration']}] "
        f"label='{entry['label']}' | "
        f"status={entry['status']} | "
        f"crashes={entry['crash_count']} | "
        f"val_auc={auc}"
    )


def _rewrite(entries):
    with open(LOG_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
