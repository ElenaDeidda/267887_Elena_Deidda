# main.py
# -----------------------------------------------------------------------------
# CLI entry point for the incremental hyperparameter search of Part 1.A.
#
# USAGE PHILOSOPHY (mirrors how the assignment's incremental greedy search is
# meant to be carried out):
#  - To find out WHICH value of a hyperparameter is best, you launch ONE GROUP
#    at a time:
#        python main.py d_model
#    Within a group, the candidate values (e.g. 128 / 256 / 384) are tried one
#    after another WITHOUT stopping; the script reports the PPL of each so you
#    can pick the best one. Unlike the other parts of this project (which
#    automate the full greedy search end-to-end in a single command), this
#    script intentionally requires the user to run one sweep group, inspect
#    the result, and then manually launch the next group with that value
#    locked in - this keeps each step auditable and lets you stop/adjust
#    between steps.
#  - Build the config up INCREMENTALLY by locking in already-chosen values via
#    command-line overrides, without editing this file. Example session:
#        python main.py d_model                      # pick, say, 384
#        python main.py n_heads     --d_model 384    # keep 384, vary heads
#        python main.py num_layers  --d_model 384 --n_heads 6
#        python main.py ff_dim      --d_model 384 --n_heads 6 --num_layers 6
#        python main.py dropout     --d_model 384 --n_heads 6 --num_layers 6 --ff_dim 1536
#        python main.py weight_tying --d_model 384 --n_heads 6 --num_layers 6 --ff_dim 1536 --dropout 0.1
#  - To try the same hyperparameter with your own custom values:
#        python main.py d_model --values 256,512,768
#  - For a single ad-hoc experiment (final config, targeted check):
#        python main.py single --d_model 384 --n_heads 6 --num_layers 6 --ff_dim 1536 \
#                              --dropout 0.1 --weight_tying --lr 3e-4
#
# Every run is appended to results/results.json and the human-readable report
# results/observations.md is regenerated automatically after each run.

import os
import argparse
import subprocess
import urllib.request

import torch

from utils import get_tokenizer, get_dataloaders
from functions import (
    run_experiment, set_seed, append_result, make_record,
    regenerate_observations_md,
)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join("dataset", "PennTreeBank")
TRAIN_PATH = os.path.join(DATA_DIR, "ptb.train.txt")
DEV_PATH = os.path.join(DATA_DIR, "ptb.valid.txt")
TEST_PATH = os.path.join(DATA_DIR, "ptb.test.txt")

RESULTS_DIR = "results"
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
OBSERVATIONS_MD = os.path.join(RESULTS_DIR, "observations.md")
BIN_DIR = "bin"

# ----------------------------------------------------------------------------
# Base configuration: the current best-known hyperparameters.
#
# This dict is the running state of the incremental greedy search: after each
# sweep group is run (e.g. `python main.py d_model`) and a winner is picked by
# inspecting dev PPL, this dict is MANUALLY updated to bake that winning value
# in as the new default, before moving on to the next group. The values below
# (d_model=384, n_heads=8, num_layers=6, ff_dim=2048, dropout=0.1,
# weight_tying=False) reflect the final winning configuration found this way,
# which achieved test PPL = 33.07 (well under the required threshold of 250).
# Command-line overrides (--d_model, --lr, ...) take precedence over this
# dict at run time (see apply_overrides), so you can also override ad hoc
# without editing the file.
# ----------------------------------------------------------------------------
BASE_CONFIG = dict(
    pos_emb_size=1024,
    d_model=384,
    n_heads=8,
    num_layers=6,
    ff_dim=2048,
    dropout=0.1,
    weight_tying=False,
)
BASE_LR = 5e-4

# ----------------------------------------------------------------------------
# Sweep definitions: group name -> (hyperparameter key, default candidate
# values to try, in order). 'lr' is handled as a special case in run_sweep
# since it is an optimizer setting, not a GPT2 model constructor argument.
#
# SWEEPS.keys() also defines the order of the incremental greedy search:
# first pick the learning rate on the baseline model (baseline-lr), then
# walk through the architecture hyperparameters one at a time (d_model,
# n_heads, num_layers, ff_dim), then the regularization choices (dropout,
# weight_tying). Each step assumes all earlier steps' winning values are
# already locked into BASE_CONFIG/passed in via CLI overrides.
# ----------------------------------------------------------------------------
SWEEPS = {
    "baseline-lr": ("lr",           [1e-3, 5e-4, 3e-4, 1e-4]),
    "d_model":     ("d_model",      [128, 256, 384]),
    "n_heads":     ("n_heads",      [2, 4, 8]),
    "num_layers":  ("num_layers",   [2, 4, 6]),
    "ff_dim":      ("ff_dim",       [512, 1024, 2048]),
    "dropout":     ("dropout",      [0.0, 0.1, 0.2]),
    "weight_tying": ("weight_tying", [False, True]),
}

GROUPS = list(SWEEPS.keys()) + ["single", "all"]


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Esperimenti incrementali GPT-2 (LM Part 1.A).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("group", choices=GROUPS,
                   help="quale gruppo di esperimenti lanciare")

    # Overrides for BASE_CONFIG's hyperparameters, used to lock in
    # already-chosen values when building the config incrementally (see the
    # module docstring's example session). Defaulting to None lets
    # apply_overrides() distinguish "not specified" from an explicit value.
    p.add_argument("--pos_emb_size", type=int)
    p.add_argument("--d_model", type=int)
    p.add_argument("--n_heads", type=int)
    p.add_argument("--num_layers", type=int)
    p.add_argument("--ff_dim", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--weight_tying", action="store_true", default=None,
                   help="attiva il weight tying nella config di base")
    p.add_argument("--lr", type=float, help="learning rate (override di BASE_LR)")

    # Custom candidate values for the sweep (e.g. --values 256,512,768),
    # overriding the defaults defined in SWEEPS for the chosen group.
    p.add_argument("--values", type=str,
                   help="lista di valori per lo sweep, separati da virgola")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=3)
    # Default changed from 8 to 32 for GPU throughput: profiling showed that
    # batch=8 produced ~5259 tiny batches/epoch, causing excessive per-batch
    # overhead (Python loop, kernel launches, optimizer step) relative to
    # actual compute, while GPU memory headroom was ample (only ~3.4GB/16GB
    # used at batch=8). A larger batch size amortizes that overhead; this is
    # a throughput change only, not part of the model-quality hyperparameter
    # search (train_bs is not swept in SWEEPS).
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_best", action="store_true",
                   help="salva lo state_dict del modello migliore di questa run in bin/")
    # --shutdown / --notify exist to support unattended runs on a rented cloud
    # GPU VM: launch a long sweep, walk away, get notified and have the VM
    # power itself off automatically (avoiding paying for idle GPU time).
    p.add_argument("--shutdown", action="store_true",
                   help="spegne la macchina al termine (anche in caso di errore)")
    p.add_argument("--notify", type=str, default=None,
                   help="topic ntfy.sh (es. 'elena-nlu-vm-x7k2') o URL/webhook "
                        "completo a cui mandare una notifica a fine run")
    return p.parse_args()


def send_notification(target, best_record=None, error=None):
    """Send a push notification when a run finishes (via ntfy.sh, or any HTTP-POST webhook).

    Used together with --notify for unattended cloud-GPU runs (see
    --shutdown), so progress/failures can be monitored remotely without
    watching the terminal.

    Args:
        target: either a bare ntfy.sh topic name (e.g. 'elena-nlu-vm-x7k2') or
            a full webhook URL to POST to.
        best_record: the best run's record dict (from make_record), if any.
        error: the exception raised during the run, if any. Takes priority
            over best_record in the message (a failed run is reported as
            such even if some results were already collected).
    """
    url = target if target.startswith("http") else f"https://ntfy.sh/{target}"

    if error is not None:
        title = "Run FALLITA"
        message = f"Errore: {type(error).__name__}: {error}"
        priority, tags = "high", "warning"
    elif best_record is not None:
        passed = best_record["passes_threshold"]
        title = f"Run finita - test PPL {best_record['test_ppl']:.2f}"
        message = (
            f"Best: {best_record['label']}\n"
            f"Test PPL: {best_record['test_ppl']:.2f} "
            f"({'OK <250' if passed else 'NON soddisfa <250'})\n"
            f"lr: {best_record['lr']}\n"
            f"config: {best_record['config']}"
        )
        priority, tags = "default", ("white_check_mark" if passed else "warning")
    else:
        title = "Run finita"
        message = "Esecuzione terminata (nessun risultato registrato)."
        priority, tags = "default", "information_source"

    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"), method="POST")
        req.add_header("Title", title)        # header HTTP: solo ASCII
        req.add_header("Priority", priority)
        req.add_header("Tags", tags)          # emoji via shortcode (es. white_check_mark)
        urllib.request.urlopen(req, timeout=10)
        print(f"[notify] notifica inviata a {url}")
    except Exception as e:
        print(f"[notify] invio fallito ({url}): {e}")


def shutdown_machine():
    """Power off the (virtual) machine, trying several shutdown commands for robustness.

    Used with --shutdown for unattended cloud GPU runs, so the VM doesn't keep
    billing/idling once the sweep is done. Tries plain and sudo-prefixed
    variants of both `shutdown` and `systemctl poweroff` since the exact
    command/permissions available vary across cloud VM images.
    """
    print("\n[shutdown] spegnimento della macchina richiesto...")
    for cmd in (["shutdown", "-h", "now"],
                ["sudo", "shutdown", "-h", "now"],
                ["systemctl", "poweroff"],
                ["sudo", "systemctl", "poweroff"]):
        try:
            subprocess.run(cmd, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("[shutdown] impossibile spegnere: nessun comando disponibile "
          "(serve probabilmente sudo senza password).")


def apply_overrides(base_cfg, base_lr, args):
    """Apply CLI overrides on top of BASE_CONFIG/BASE_LR.

    This is the mechanism that makes the incremental search possible without
    editing main.py: any hyperparameter explicitly passed on the command line
    (e.g. --d_model 384) takes precedence over BASE_CONFIG's current value,
    letting you "lock in" previously chosen values while sweeping the next one.
    """
    cfg = dict(base_cfg)
    for key in ["pos_emb_size", "d_model", "n_heads", "num_layers", "ff_dim", "dropout"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.weight_tying:
        cfg["weight_tying"] = True
    lr = args.lr if args.lr is not None else base_lr
    return cfg, lr


def parse_values(raw, key):
    """Parse the --values CLI string into the correct Python type for the given sweep key."""
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if key in ("lr", "dropout"):
        return [float(s) for s in parts]
    if key == "weight_tying":
        return [s.lower() in ("1", "true", "yes", "si", "sì") for s in parts]
    return [int(s) for s in parts]


def run_sweep(group, key, values, base_cfg, base_lr, loaders, tokenizer, device, args):
    """Run every candidate value of one sweep group, one after another, logging each.

    This is the core of the "one group at a time" workflow: for the chosen
    group (e.g. "d_model"), it builds a config for each candidate value
    (locking everything else to base_cfg/base_lr), runs a full experiment via
    run_experiment, and appends each result to results/results.json,
    regenerating the Markdown report after every single run (so partial
    progress is visible even if the sweep is interrupted).

    Returns:
        (best_model, best_record): the run with the lowest dev PPL in this
        group (selection is on dev PPL, never on test PPL - see below).
    """
    print(f"\n########## GRUPPO '{group}' | varia '{key}' su {values} ##########")
    best_model, best_record = None, None

    for val in values:
        cfg = dict(base_cfg)
        lr = base_lr
        if key == "lr":
            lr = val
        else:
            cfg[key] = val

        # Skip invalid combinations (d_model not divisible by n_heads) rather
        # than letting MultiHeadAttention's assertion crash the whole sweep.
        if cfg["d_model"] % cfg["n_heads"] != 0:
            print(f"[skip] d_model={cfg['d_model']} non divisibile per "
                  f"n_heads={cfg['n_heads']}: combinazione saltata.")
            continue

        label = f"{group}={val}"
        set_seed(args.seed)  # same seed for every run -> fair, controlled comparison
        model, info = run_experiment(
            label, cfg, loaders, tokenizer, device,
            lr=lr, n_epochs=args.epochs, patience=args.patience,
        )

        record = make_record(group, label, info,
                             swept_key=key, swept_value=val, seed=args.seed)
        append_result(RESULTS_JSON, record)
        regenerate_observations_md(RESULTS_JSON, OBSERVATIONS_MD)

        # Best-of-group selection is ALWAYS on dev PPL (info["best_dev_ppl"]),
        # never on test PPL, consistent with run_experiment's own selection
        # policy - keeps the test set strictly for final reporting only.
        if best_record is None or info["best_dev_ppl"] < best_record["best_dev_ppl"]:
            best_model, best_record = model, record

    if best_record is not None:
        print(f"\n>>> Migliore del gruppo '{group}' (scelto su dev): {best_record['label']} "
              f"-> dev PPL {best_record['best_dev_ppl']:.2f} | test PPL {best_record['test_ppl']:.2f}")
    return best_model, best_record


def main():
    """CLI entry point: run the requested sweep/experiment, then notify/shutdown if requested.

    The try/except/finally structure guarantees that --notify and --shutdown
    still fire even if training raises an exception partway through (e.g. an
    OOM on a late sweep value), which matters for unattended cloud GPU runs
    where nobody is watching the terminal to notice a crash.
    """
    args = parse_args()
    best_record = None
    err = None
    try:
        best_record = _run(args)
    except Exception as e:
        err = e
    finally:
        # Correct order: training (already done above) -> notify -> shutdown.
        # Always executed, so it notifies/shuts down even if training errored.
        if args.notify:
            send_notification(args.notify, best_record=best_record, error=err)
        if args.shutdown:
            shutdown_machine()
    # Re-raise any error AFTER notifying/shutting down, so the process still
    # exits with a correct non-zero exit code and the traceback ends up in
    # the logs (useful for diagnosing failed unattended runs after the fact).
    if err is not None:
        raise err


def _run(args):
    """Set up data/device and dispatch to the requested group: single run, full sweep, or 'all'."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = get_tokenizer()
    loaders = get_dataloaders(
        TRAIN_PATH, DEV_PATH, TEST_PATH,
        tokenizer=tokenizer, device=device,
        train_bs=args.train_bs, eval_bs=args.eval_bs,
    )

    base_cfg, base_lr = apply_overrides(BASE_CONFIG, BASE_LR, args)
    print(f"Config di base (dopo override): {base_cfg} | lr: {base_lr}")

    best_model, best_record = None, None

    if args.group == "single":
        # One single experiment using the base config (plus any overrides) -
        # used for ad-hoc/final-config runs outside the sweep machinery.
        set_seed(args.seed)
        model, info = run_experiment(
            "single", base_cfg, loaders, tokenizer, device,
            lr=base_lr, n_epochs=args.epochs, patience=args.patience,
        )
        record = make_record("single", "single", info, seed=args.seed)
        append_result(RESULTS_JSON, record)
        regenerate_observations_md(RESULTS_JSON, OBSERVATIONS_MD)
        best_model, best_record = model, record

    elif args.group == "all":
        # Convenience mode: runs every sweep group back-to-back on the SAME
        # base config. NOTE: this is NOT the incremental greedy search (that
        # requires running one group at a time and manually feeding the
        # winning value forward as an override) - it's just a broad overview
        # of how each hyperparameter behaves around the current base config.
        for g, (key, default_values) in SWEEPS.items():
            values = parse_values(args.values, key) if args.values else default_values
            m, r = run_sweep(g, key, values, base_cfg, base_lr,
                             loaders, tokenizer, device, args)
            if r is not None and (best_record is None or r["best_dev_ppl"] < best_record["best_dev_ppl"]):
                best_model, best_record = m, r

    else:
        # Single sweep group - the main use case for the incremental search.
        key, default_values = SWEEPS[args.group]
        values = parse_values(args.values, key) if args.values else default_values
        best_model, best_record = run_sweep(
            args.group, key, values, base_cfg, base_lr,
            loaders, tokenizer, device, args,
        )

    # Optionally persist the best model's weights from this run to disk.
    if args.save_best and best_model is not None:
        os.makedirs(BIN_DIR, exist_ok=True)
        save_path = os.path.join(BIN_DIR, "best_model.pt")
        best_model.to("cpu")
        torch.save(best_model.state_dict(), save_path)
        print(f"\nModello migliore salvato in: {save_path}")
        print(f"Per ricaricarlo usa la config: {best_record['config']}")

    print(f"\nRisultati JSON: {RESULTS_JSON}")
    print(f"Report osservazioni: {OBSERVATIONS_MD}")

    return best_record


if __name__ == "__main__":
    main()
