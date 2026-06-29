# main.py
# Part 1.B - Fine-tuning a pre-trained GPT-2 with LoRA on the Penn TreeBank corpus.
#
# Only the LoRA matrices on the query/key/value projections are trained (roughly
# 0.2-1% of all parameters depending on rank); the ~124M original GPT-2 weights stay
# frozen throughout.
#
# AUTOMATIC GREEDY HYPERPARAMETER SEARCH (one hyperparameter group at a time, winner of
# each step chosen strictly by DEV PPL), all run in a single command -- unlike Part
# 1.A, which required separate manual invocations per step:
#   Step 0 - learning rate            (rank=4, alpha=8 held fixed)
#   Step 1 - LoRA rank r in {4, 8, 16} (alpha = 2*r -> scaling fixed at 2.0)
#   Step 2 - alpha, at the winning rank from step 1 (scaling = alpha/rank varies)
#
# The winning configuration of each step is automatically merged into the running
# `config` dict and carried forward into the next step (see run_search). Every
# individual run is appended to results/results.json; the search is idempotent because
# experiments whose name is already present in that file are skipped on a re-run (see
# done_experiments/get_or_run below), which makes it safe to resume after an
# interruption (e.g. a VM reboot triggered by --shutdown) without duplicating
# already-completed work. results/observations.md is regenerated after every run.
#
# Usage:
#   python main.py                         # runs the full greedy search end-to-end
#   python main.py --epochs 20 --seed 42
#   python main.py --notify <ntfy_topic> --shutdown   # for unattended cloud GPU runs
#
# Target: test PPL < 250 (mandatory) and better than Part 1.A's from-scratch baseline.

import os
import gc
import argparse
import random
import subprocess
import urllib.request

import torch

from utils import read_file, get_tokenizer, get_dataloaders
from model import GPT2_LoRA, param_stats
from functions import (
    freeze_pretrained_and_enable_lora, train_model, eval_loop, count_trainable,
    append_result, make_record, done_experiments, regenerate_observations_md,
    load_results, PPL_THRESHOLD,
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
# Starting config and greedy search plan
# ----------------------------------------------------------------------------
BASE = {"rank": 4, "alpha": 8, "lr": 5e-4, "batch_size": 8}

# SEARCH_STEPS: a list of (step_index, [(experiment_name, override_dict), ...]) tuples
# consumed in order by run_search. For each step, every candidate's override_dict is
# layered on top of the *current* `config` (itself the winner propagated from the
# previous step -- see run_search), so step 1 already runs at the winning learning
# rate from step 0, and step 2 already runs at the winning rank from step 1. Step 2's
# candidates use the special "alpha_factor" key instead of an explicit "alpha" value
# because alpha must be expressed relative to whichever rank step 1 selected (which is
# only known at runtime, not when this list is written).
SEARCH_STEPS = [
    # Step 0 - learning rate search (rank=4, alpha=8 held fixed)
    (0, [
        ("step0_lr1e-3", {"lr": 1e-3}),
        ("step0_lr5e-4", {"lr": 5e-4}),
        ("step0_lr1e-4", {"lr": 1e-4}),
    ]),
    # Step 1 - LoRA rank search (alpha = 2*rank -> scaling fixed at 2.0 for all three,
    # isolating the effect of rank alone)
    (1, [
        ("step1_rank4",  {"rank": 4,  "alpha": 8}),
        ("step1_rank8",  {"rank": 8,  "alpha": 16}),
        ("step1_rank16", {"rank": 16, "alpha": 32}),
    ]),
    # Step 2 - alpha search at the best rank found in step 1 (scaling = alpha/rank
    # varies across these three candidates)
    (2, [
        ("step2_alpha_half", {"alpha_factor": 0.5}),   # alpha = rank/2
        ("step2_alpha_eq",   {"alpha_factor": 1.0}),   # alpha = rank
        ("step2_alpha_2x",   {"alpha_factor": 2.0}),   # alpha = 2*rank
    ]),
]


def parse_args():
    """Define and parse the command-line interface for this script."""
    p = argparse.ArgumentParser(description="Fine-tuning LoRA di GPT-2 (LM Part 1.B).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline_ppl", type=float, default=None,
                   help="miglior test PPL della Part 1.A, solo per il report observations.md")
    p.add_argument("--no_save", action="store_true",
                   help="non salvare i pesi LoRA del modello migliore in bin/")
    p.add_argument("--notify", type=str, default=None,
                   help="topic ntfy.sh o URL webhook per la notifica a fine run")
    p.add_argument("--shutdown", action="store_true",
                   help="spegne la macchina al termine")
    return p.parse_args()


def set_seed(seed=42):
    """Seed Python's, PyTorch's and (if available) CUDA's RNGs for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Pick the best available compute device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def send_notification(target, best=None, error=None):
    """Send a push notification when the run finishes (success or failure).

    `target` is either a full webhook URL or an ntfy.sh topic name; useful for
    unattended multi-hour runs on a cloud GPU instance where polling the terminal
    isn't practical.
    """
    url = target if target.startswith("http") else f"https://ntfy.sh/{target}"
    if error is not None:
        title = "Run 1.B FALLITA"
        message = f"Errore: {type(error).__name__}: {error}"
    elif best is not None:
        title = f"Run 1.B finita - test PPL {best['test_ppl']:.2f}"
        message = (f"Best: {best['experiment']}\n"
                   f"dev PPL {best['best_dev_ppl']:.2f} | test PPL {best['test_ppl']:.2f}\n"
                   f"rank={best['rank']} alpha={best['alpha']} lr={best['lr']}")
    else:
        title, message = "Run 1.B finita", "Nessun risultato registrato."
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"), method="POST")
        req.add_header("Title", title)
        urllib.request.urlopen(req, timeout=10)
        print(f"[notify] notifica inviata a {url}")
    except Exception as e:
        print(f"[notify] invio fallito ({url}): {e}")


def shutdown_machine():
    """Power off the host machine, trying a few common shutdown command variants in
    turn. Intended for unattended cloud GPU runs (--shutdown flag) so the instance
    doesn't keep billing/running after the multi-hour search completes."""
    print("\n[shutdown] spegnimento della macchina richiesto...")
    for cmd in (["sudo", "shutdown", "-h", "now"], ["shutdown", "-h", "now"],
                ["sudo", "systemctl", "poweroff"], ["systemctl", "poweroff"]):
        try:
            subprocess.run(cmd, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("[shutdown] impossibile spegnere (serve sudo senza password).")


def main():
    """Entry point: run the full greedy search, then notify/shutdown as requested
    regardless of whether the search succeeded or raised (the notify/shutdown calls
    live in `finally` so they fire even on failure) -- and finally re-raise any
    exception so the script still exits with a non-zero status on error."""
    args = parse_args()
    best_record = None
    err = None
    try:
        best_record = run_search(args)
    except Exception as e:
        err = e
    finally:
        if args.notify:
            send_notification(args.notify, best=best_record, error=err)
        if args.shutdown:
            shutdown_machine()
    if err is not None:
        raise err


def run_search(args):
    """Run the full 3-step greedy hyperparameter search end-to-end.

    For each step in SEARCH_STEPS, every candidate is trained and evaluated (or
    fetched from a previous run via get_or_run's idempotent skip logic), the candidate
    with the lowest dev PPL is selected as that step's winner, and the winner's
    rank/alpha/lr are merged into `config` before moving to the next step. After all
    steps complete, the overall best configuration (by dev PPL, across every step) is
    optionally re-trained once more solely to persist its LoRA adapter weights to disk.
    """
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(BIN_DIR, exist_ok=True)

    tokenizer = get_tokenizer()
    train_raw = read_file(TRAIN_PATH)
    dev_raw = read_file(DEV_PATH)
    test_raw = read_file(TEST_PATH)
    print(f"Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}")
    print(f"Target: PPL test < {PPL_THRESHOLD} e migliore della Part 1.A\n")

    done = done_experiments(RESULTS_JSON)
    records = {r["experiment"]: r for r in load_results(RESULTS_JSON)}

    def get_or_run(name, step, cfg):
        """Return the result record for experiment `name`/`cfg`: read it back from the
        JSON results file if it's already there, otherwise actually train+evaluate it
        and append the new record.

        This is the idempotency mechanism that makes the whole search safely
        resumable: `done` is the set of experiment names already present in
        results.json (computed once at the top of run_search), so re-running this
        script after an interruption (crash, VM reboot from --shutdown, manual Ctrl-C)
        will skip every experiment that already has a saved result and only continue
        from the first one that doesn't -- without ever duplicating a completed run.
        """
        if name in done:
            print(f"[skip] {name} (gia' in {RESULTS_JSON})")
            return records[name]
        rec = run_one(name, step, cfg, tokenizer, train_raw, dev_raw, test_raw,
                      device, args, save_lora=False)
        append_result(RESULTS_JSON, rec)
        regenerate_observations_md(RESULTS_JSON, OBSERVATIONS_MD, baseline_ppl=args.baseline_ppl)
        done.add(name)
        records[name] = rec
        return rec

    # Greedy cascade: at each step, every candidate is layered on top of `config` (the
    # winner carried over from the previous step), and only the parameters changed by
    # the step's winner are written back into `config` before the next step begins.
    config = dict(BASE)
    best_record = None

    for step, candidates in SEARCH_STEPS:
        print(f"\n########## STEP {step} ##########")
        step_best = None
        for name, override in candidates:
            trial = dict(config)
            # Step 2 expresses alpha as a multiplicative factor of the *current*
            # (already-decided) rank rather than an absolute value, since the winning
            # rank is only known once step 1 has finished.
            if "alpha_factor" in override:
                trial["alpha"] = int(round(config["rank"] * override["alpha_factor"]))
            trial.update({k: v for k, v in override.items() if k != "alpha_factor"})

            rec = get_or_run(name, step, trial)
            d = rec["best_dev_ppl"]
            print(f"  {name}: rank={trial['rank']} alpha={trial['alpha']} "
                  f"lr={trial['lr']} -> dev PPL {d:.2f}")
            if step_best is None or d < step_best["best_dev_ppl"]:
                step_best = rec
        # Propagate this step's winner (selected strictly by dev PPL, never test PPL)
        # into `config` so every subsequent step builds on top of it.
        config["rank"] = step_best["rank"]
        config["alpha"] = step_best["alpha"]
        config["lr"] = step_best["lr"]
        if best_record is None or step_best["best_dev_ppl"] < best_record["best_dev_ppl"]:
            best_record = step_best
        print(f"  -> vincitore step {step}: {step_best['experiment']} "
              f"(dev PPL {step_best['best_dev_ppl']:.2f})")

    print("\n" + "=" * 70)
    print(f"CONFIG MIGLIORE (selezionata sulla DEV PPL): {best_record['experiment']}")
    print(f"  dev PPL {best_record['best_dev_ppl']:.2f} | test PPL {best_record['test_ppl']:.2f} "
          f"| rank={best_record['rank']} alpha={best_record['alpha']} lr={best_record['lr']}")
    print("=" * 70)

    # Re-train the overall best configuration one more time, purely to obtain a model
    # instance whose LoRA weights can be persisted to bin/ (none of the per-step runs
    # above save weights, to avoid storing many near-duplicate checkpoints during the
    # search itself).
    if not args.no_save:
        best_pt = os.path.join(BIN_DIR, f"{best_record['experiment']}_lora.pt")
        if not os.path.exists(best_pt):
            print(f"\n[bin] Ri-alleno '{best_record['experiment']}' per salvarlo in {best_pt} ...")
            run_one(best_record["experiment"], "best",
                    {"rank": best_record["rank"], "alpha": best_record["alpha"],
                     "lr": best_record["lr"], "batch_size": best_record["batch_size"]},
                    tokenizer, train_raw, dev_raw, test_raw, device, args,
                    save_lora=True, save_path=best_pt)
        else:
            print(f"\n[bin] Modello migliore gia' presente: {best_pt}")

    print(f"\nRisultati JSON: {RESULTS_JSON}")
    print(f"Report osservazioni: {OBSERVATIONS_MD}")
    return best_record


def run_one(name, step, cfg, tokenizer, train_raw, dev_raw, test_raw,
            device, args, save_lora=False, save_path=None):
    """Run a single LoRA experiment end-to-end: load pre-trained GPT-2, inject LoRA
    adapters at the requested rank/alpha, freeze the backbone, train with early
    stopping on dev PPL, evaluate on test, and optionally persist the LoRA weights.
    Returns the JSON-serializable result record for this run (see make_record)."""
    print(f"\n===== Esperimento: {name} =====")
    print(f"  rank={cfg['rank']} alpha={cfg['alpha']} "
          f"scaling={cfg['alpha']/cfg['rank']:.2f} lr={cfg['lr']} bs={cfg['batch_size']}")
    set_seed(args.seed)

    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, tokenizer, device,
        batch_size_train=cfg["batch_size"], batch_size_eval=cfg["batch_size"] * 2,
    )

    # from_pretrained constructs a GPT2_LoRA (injecting fresh, randomly-initialized
    # LoRA adapters per CustomGPT2Attention.__init__) and then loads the pre-trained
    # GPT-2 checkpoint into every non-LoRA parameter -- see model.py for the exact
    # init/load ordering and why it leaves the LoRA matrices untouched.
    model = GPT2_LoRA.from_pretrained(
        "openai-community/gpt2", rank=cfg["rank"], alpha=cfg["alpha"],
    )
    model.to(device)
    freeze_pretrained_and_enable_lora(model)  # only lora_* parameters keep requires_grad=True
    param_stats(model)

    # Restricting the optimizer to `p.requires_grad` parameters means AdamW only ever
    # sees (and only ever updates) the LoRA matrices -- the frozen backbone parameters
    # are never even registered with the optimizer.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["lr"], weight_decay=0.01,
    )

    best_model, best_dev_ppl, best_epoch, epochs_run = train_model(
        model, train_loader, dev_loader, tokenizer, optimizer,
        n_epochs=args.epochs, patience=args.patience, experiment_name=name,
    )

    best_model = best_model.to(device)
    test_ppl, test_loss = eval_loop(test_loader, best_model, tokenizer)
    print(f"  [RISULTATI] {name}: dev PPL {best_dev_ppl:.2f} | test PPL {test_ppl:.2f} "
          f"| <250: {'OK' if test_ppl < PPL_THRESHOLD else 'NO'}")

    if save_lora and save_path is not None:
        # Save ONLY the LoRA adapter tensors (state_dict keys containing "lora_"),
        # filtering out the ~124M-parameter pre-trained backbone entirely. This keeps
        # the checkpoint on the order of a few MB instead of several hundred MB,
        # since the backbone is never modified and can always be re-obtained from
        # HuggingFace (openai-community/gpt2) -- there is no reason to duplicate it
        # on disk for every saved experiment.
        lora_state = {k: v for k, v in best_model.state_dict().items() if "lora_" in k}
        torch.save(lora_state, save_path)
        print(f"  Pesi LoRA salvati in: {save_path} ({len(lora_state)} tensori)")

    info = {
        "rank": cfg["rank"], "alpha": cfg["alpha"], "lr": cfg["lr"],
        "batch_size": cfg["batch_size"], "n_trainable": count_trainable(model),
        "best_epoch": best_epoch, "epochs_run": epochs_run,
        "best_dev_ppl": best_dev_ppl, "test_ppl": test_ppl, "test_loss": test_loss,
    }
    record = make_record(name, step, info, seed=args.seed)

    # pulizia memoria
    del model, best_model, optimizer, train_loader, dev_loader, test_loader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return record


if __name__ == "__main__":
    main()
