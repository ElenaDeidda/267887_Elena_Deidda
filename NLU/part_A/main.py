# main.py
# Part 2.A - NLU from scratch (Intent Classification + Slot Filling) on ATIS.
#
# Entry point that runs the FULLY AUTOMATIC greedy hyperparameter search for
# the from-scratch GPT-2 model, in a single command. Each candidate
# configuration is repeated over multiple runs (mean +- std of conll chunk-
# level Slot F1 and Intent Accuracy) before being compared to the running
# best ("incumbent").
#
# GREEDY SEARCH (one hyperparameter group at a time, winner picked on DEV F1):
#   Step 0 - learning rate
#   Step 1 - architecture: d_model, n_heads, num_layers, ff_dim (each tested
#            independently against the SAME incumbent score from step 0 -
#            these are NOT chained to each other within step 1; see
#            run_search below for exactly how the incumbent carries over)
#   Step 2 - dropout (including the dropout applied right before the output
#            heads)
#
# A step's winning value becomes part of the config used by the NEXT step,
# but a step can also end with NO change at all if none of its candidates'
# mean dev F1 strictly exceeds the incumbent's (this is exactly what happened
# for d_model, num_layers, ff_dim and dropout in the reported final run -
# only n_heads improved on the baseline). Results are cached by experiment
# name in results/results.json (idempotent: a finished experiment is never
# retrained) and summarized in observations.md.
#
# Usage:
#   python main.py
#   python main.py --runs 5 --epochs 200
#   python main.py --notify <topic_ntfy> --shutdown   # convenient on a cloud VM

import os
import argparse
import random
import subprocess
import urllib.request

import torch
import torch.nn as nn

from utils import (load_data, create_dev_split, build_lang,
                   IntentsAndSlots, get_dataloaders, PAD_TOKEN)
from model import GPT2, init_weights
from functions import (
    run_experiments, train_model,
    append_result, make_record, done_experiments, load_results,
    regenerate_observations_md,
)

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join("dataset", "ATIS")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")

RESULTS_DIR = "results"
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
OBSERVATIONS_MD = os.path.join(RESULTS_DIR, "observations.md")
BIN_DIR = "bin"

# ----------------------------------------------------------------------------
# Starting (baseline) config and greedy search plan (one hyperparameter group
# at a time)
# ----------------------------------------------------------------------------
BASE = {"lr": 0.01, "d_model": 64, "n_heads": 2,
        "num_layers": 2, "ff_dim": 256, "dropout": 0.0}

# List of (param_name, file_label_prefix, [candidate_values]) tuples, one per
# search step, processed in order by run_search. The incumbent value (i.e.
# config[param] as it stands when the step starts) is NOT itself retrained
# here -- its score is already known from a previous step (or, for the very
# first step's parameter, is implicitly absent and handled via
# `best_score is None` in run_search) and is simply used as the bar each new
# candidate must strictly clear.
SEARCH_STEPS = [
    ("lr",         "step0_lr",     [0.1, 0.01, 0.001, 0.0001]),
    ("d_model",    "step1_dmodel", [128, 256]),
    ("n_heads",    "step1_nheads", [4, 8]),
    ("num_layers", "step1_layers", [4, 6]),
    ("ff_dim",     "step1_ffdim",  [512, 1024]),
    ("dropout",    "step2_dropout", [0.1, 0.2, 0.3]),
]


def parse_args():
    """Define and parse the command-line interface (see module docstring for
    example invocations)."""
    p = argparse.ArgumentParser(description="NLU intent+slot from scratch (Part 2.A).")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_save", action="store_true",
                   help="non salvare il modello migliore in bin/")
    p.add_argument("--notify", type=str, default=None,
                   help="topic ntfy.sh o URL webhook per la notifica a fine run")
    p.add_argument("--shutdown", action="store_true", help="spegne la macchina al termine")
    return p.parse_args()


def set_seed(seed=42):
    """Seed Python's random, torch CPU and (if available) all CUDA devices,
    for reproducibility of the single final "save the best model" run."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Pick the best available compute device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def send_notification(target, best=None, error=None):
    """POST a push notification (via ntfy.sh or a custom webhook URL) when the
    search finishes or fails. Convenience for unattended runs on a remote VM;
    failures to notify are caught and logged, never raised."""
    url = target if target.startswith("http") else f"https://ntfy.sh/{target}"
    if error is not None:
        title, message = "Run 2.A FALLITA", f"Errore: {type(error).__name__}: {error}"
    elif best is not None:
        title = f"Run 2.A finita - test F1 {best['slot_f1_mean']:.3f}"
        message = (f"Best: {best['experiment']}\n"
                   f"dev F1 {best['dev_f1_mean']:.3f} | test F1 {best['slot_f1_mean']:.3f} "
                   f"| test acc {best['intent_acc_mean']:.3f}")
    else:
        title, message = "Run 2.A finita", "Nessun risultato registrato."
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"), method="POST")
        req.add_header("Title", title)
        urllib.request.urlopen(req, timeout=10)
        print(f"[notify] notifica inviata a {url}")
    except Exception as e:
        print(f"[notify] invio fallito ({url}): {e}")


def shutdown_machine():
    """Attempt to power off the host machine (with or without sudo, depending
    on what's available) once the search completes; used with --shutdown on
    a disposable cloud VM to avoid paying for idle time."""
    print("\n[shutdown] spegnimento della macchina richiesto...")
    for cmd in (["sudo", "shutdown", "-h", "now"], ["shutdown", "-h", "now"],
                ["sudo", "systemctl", "poweroff"], ["systemctl", "poweroff"]):
        try:
            subprocess.run(cmd, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("[shutdown] impossibile spegnere (serve sudo senza password).")


def setup_data(device):
    """Load ATIS train/test, carve out the dev split, build the Lang
    vocabularies, and construct the train/dev/test DataLoaders."""
    tmp_train_raw = load_data(TRAIN_PATH)
    test_raw = load_data(TEST_PATH)
    print(f"Training set originale: {len(tmp_train_raw)} | Test set: {len(test_raw)}")

    train_raw, dev_raw = create_dev_split(tmp_train_raw, dev_size=0.10)
    print(f"Training (dopo split): {len(train_raw)} | Dev set: {len(dev_raw)}")

    lang = build_lang(train_raw, dev_raw, test_raw, cutoff=0)
    print(f"Vocabolario: {len(lang.word2id)} parole | "
          f"Slot: {len(lang.id2slot)} | Intent: {len(lang.intent2id)}")

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    loaders = get_dataloaders(train_dataset, dev_dataset, test_dataset,
                              batch_size_train=128, batch_size_eval=64, device=device)
    dims = (len(lang.word2id), len(lang.id2slot), len(lang.intent2id))
    return loaders, lang, dims


def train_and_save_best(cfg, loaders, lang, dims, device, path, args):
    """Train a SINGLE instance of the winning configuration (one seeded run,
    not averaged like run_experiments) and persist its state_dict, config,
    and vocabularies to `path` (under bin/), so the final model can be
    reloaded later for inference without rerunning the whole search."""
    train_loader, dev_loader, _ = loaders
    vocab_len, slots_len, n_intents = dims

    set_seed(args.seed)
    model = GPT2(vocab_size=vocab_len, slots_size=slots_len, n_intents=n_intents,
                 pos_emb_size=1024, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                 num_layers=cfg["num_layers"], ff_dim=cfg["ff_dim"], dropout=cfg["dropout"]).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    best_model, best_f1, best_acc = train_model(
        model, train_loader, dev_loader, lang, optimizer,
        criterion_slots, criterion_intents,
        n_epochs=args.epochs, patience=args.patience, experiment_name=f"{cfg['experiment']}_BEST",
    )
    torch.save({"model": best_model.state_dict(), "config": cfg,
                "word2id": lang.word2id, "slot2id": lang.slot2id,
                "intent2id": lang.intent2id}, path)
    print(f"  Modello migliore salvato in: {path} (dev F1={best_f1:.4f}, dev Acc={best_acc:.4f})")


def main():
    """CLI entry point: run the full greedy search and, regardless of
    success or failure, optionally send a notification and/or shut the
    machine down (both intended for unattended runs on a cloud VM). Any
    exception from run_search is re-raised after the finally-block
    notify/shutdown housekeeping, so the process still exits with an error
    status on failure."""
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
    """Run the full greedy hyperparameter search described in the module
    docstring (Step 0 lr, Step 1 architecture, Step 2 dropout) and, unless
    --no_save was passed, train and persist one final instance of the
    overall winning configuration.

    Returns the JSON record (dict) of the best experiment found, selected by
    DEV slot F1 (see make_record / run_experiments).
    """
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(BIN_DIR, exist_ok=True)

    loaders, lang, dims = setup_data(device)
    train_loader, dev_loader, test_loader = loaders
    vocab_len, slots_len, n_intents = dims

    done = done_experiments(RESULTS_JSON)
    records = {r["experiment"]: r for r in load_results(RESULTS_JSON)}

    def get_or_run(name, param, cfg):
        """Return the experiment record for `cfg` under name `name`: loaded
        from the JSON cache if `name` is already in `done` (idempotent skip,
        so re-running main.py after an interruption does not retrain
        anything already completed), otherwise trained from scratch via
        run_experiments (args.runs seeds) and then appended to the JSON
        cache / observations.md before being returned."""
        if name in done:
            print(f"[skip] {name} (gia' in {RESULTS_JSON})")
            return records[name]
        info = run_experiments(
            train_loader, dev_loader, test_loader, lang,
            vocab_len, slots_len, n_intents,
            lr=cfg["lr"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            num_layers=cfg["num_layers"], ff_dim=cfg["ff_dim"], dropout=cfg["dropout"],
            n_runs=args.runs, n_epochs=args.epochs, patience=args.patience,
            experiment_name=name, seed=args.seed, device=device,
        )
        rec = make_record(name, param, info, seed=args.seed)
        append_result(RESULTS_JSON, rec)
        regenerate_observations_md(RESULTS_JSON, OBSERVATIONS_MD)
        done.add(name)
        records[name] = rec
        return rec

    # Greedy cascade: at every step, the running-best ("incumbent") config is
    # carried forward and only replaced if a candidate strictly beats it.
    config = dict(BASE)
    best_score = None   # dev F1 of the current incumbent (None before step 0 has a result)
    best_record = None

    for param, label, candidates in SEARCH_STEPS:
        print(f"\n########## STEP '{param}' (incumbent = {config[param]}) ##########")
        # IMPORTANT: step_value/step_score/step_rec are initialized from the
        # CURRENT incumbent (config[param] and the best_score/best_record
        # carried over from the previous step) BEFORE any candidate in this
        # step is evaluated. This is what implements "test independently
        # against the running-best incumbent, not chained": if NONE of this
        # step's candidates achieve a dev F1 strictly greater than
        # step_score, the loop below never updates step_value, so
        # config[param] is reassigned to the same value it already had and
        # the parameter is effectively left unchanged for all subsequent
        # steps. This is exactly what happens for d_model, num_layers,
        # ff_dim, and dropout in the reported run: their candidates all
        # score below the incumbent dev F1 already achieved earlier (e.g.
        # with d_model=64 from the lr step), so each of those parameters
        # keeps its BASE value through the rest of the search.
        step_value, step_score, step_rec = config[param], best_score, best_record
        for v in candidates:
            name = f"{label}{v}"
            rec = get_or_run(name, param, {**config, param: v})
            score = rec["dev_f1_mean"]
            print(f"  {name}: dev F1 = {score:.4f} (test F1 {rec['slot_f1_mean']:.4f})")
            # Strict ">" (not ">="): a candidate must genuinely beat the
            # incumbent to be adopted, so ties keep the simpler/incumbent value.
            if step_score is None or score > step_score:
                step_value, step_score, step_rec = v, score, rec
        config[param] = step_value
        best_score, best_record = step_score, step_rec
        print(f"  -> '{param}' = {step_value} (dev F1 = {step_score:.4f})")

    print("\n" + "=" * 70)
    print(f"CONFIG MIGLIORE (selezionata sulla DEV F1): {best_record['experiment']}")
    print(f"  dev F1 {best_record['dev_f1_mean']:.4f} | test Slot F1 {best_record['slot_f1_mean']:.4f} "
          f"| test Intent Acc {best_record['intent_acc_mean']:.4f}")
    print(f"  {config}")
    print("=" * 70)

    if not args.no_save:
        best_pt = os.path.join(BIN_DIR, f"{best_record['experiment']}.pt")
        if not os.path.exists(best_pt):
            print(f"\n[bin] Alleno '{best_record['experiment']}' (1 run) per salvarlo in {best_pt} ...")
            train_and_save_best({"experiment": best_record["experiment"], **config},
                                loaders, lang, dims, device, best_pt, args)
        else:
            print(f"\n[bin] Modello migliore gia' presente: {best_pt}")

    print(f"\nRisultati JSON: {RESULTS_JSON}")
    print(f"Report osservazioni: {OBSERVATIONS_MD}")
    return best_record


if __name__ == "__main__":
    main()
