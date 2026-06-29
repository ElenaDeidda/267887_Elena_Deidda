# This is the entry point and experiment orchestrator for Part 2.B. For each
# model family (bert: bert-base-uncased -> bert-large-uncased; gpt2:
# openai-community/gpt2 -> gpt2-medium) it runs, in a single command, a fully
# automatic 2-step GREEDY hyperparameter search on the BASE variant only:
#   Step 0: try 3 learning-rate candidates, keep the best (by mean dev Slot F1
#           across --runs seeds, to avoid seed-luck bias).
#   Step 1: with that winning lr fixed, try 2 dropout candidates, keep the best.
# The winning (lr, dropout) config from this base-model search is then reused,
# WITHOUT being re-searched, to fine-tune the family's larger variant
# (bert-large-uncased / gpt2-medium) exactly once. This base-search-then-
# evaluate-large pattern is a deliberate, cost-driven design choice: an
# exhaustive lr/dropout search repeated on the ~3x larger model would have
# been far more expensive, while reusing the base model's winning config still
# gives a meaningful, fair-ish comparison point to check whether scaling up
# helps (empirically here, it does not: test metrics for bert-large/gpt2-medium
# do not improve over bert-base/gpt2-base despite the parameter increase).
#
# The greedy search and the base->large hand-off are both implemented with the
# same idempotent skip-if-done pattern used throughout this project: every
# experiment is named deterministically, looked up in results/results.json
# before running, and skipped (with the logged record reused) if already
# present — so re-running `python main.py` after an interruption resumes
# rather than redoing finished work. The incumbent config dict is mutated
# in-place as each search step's winner is found (config[param] = step_value),
# so later steps and the final large/medium run automatically see the
# winning values of all earlier steps.
#
# bert-base/bert-large share one WordPiece vocabulary, and gpt2/gpt2-medium
# share one BPE vocabulary, so the DataLoaders (and thus the tokenization /
# label alignment performed in utils.py) built once for the base model are
# reused unchanged for the large/medium model within the same family.

import os
import argparse
import random
import subprocess
import urllib.request

import torch
import torch.nn as nn

from utils import (load_data, create_dev_split, build_lang,
                   BERTIntentsAndSlots, GPT2IntentsAndSlots,
                   collate_fn_bert, collate_fn_gpt2, get_dataloaders,
                   get_bert_tokenizer, get_gpt2_tokenizer, IGNORE_INDEX)
from model import BERTforNLU, GPT2forNLU
from functions import (
    run_experiments, train_model,
    append_result, make_record, done_experiments, load_results,
    regenerate_observations_md,
)

# ----------------------------------------------------------------------------
# Percorsi
# ----------------------------------------------------------------------------
DATA_DIR        = os.path.join("dataset", "ATIS")
TRAIN_PATH      = os.path.join(DATA_DIR, "train.json")
TEST_PATH       = os.path.join(DATA_DIR, "test.json")

RESULTS_DIR     = "results"
RESULTS_JSON    = os.path.join(RESULTS_DIR, "results.json")
OBSERVATIONS_MD = os.path.join(RESULTS_DIR, "observations.md")
BIN_DIR         = "bin"

# ----------------------------------------------------------------------------
# Famiglie di modelli: base -> large/medium (stesso tokenizer per entrambi)
# ----------------------------------------------------------------------------
# Each family maps its "base" variant (subject to the full greedy search) to
# its "large"/"medium" variant (fine-tuned once, reusing the base's winning
# config — see module docstring above). "type" selects which dataset/model
# classes and forward signature (model_type in model.py/functions.py) apply.
MODEL_FAMILIES = {
    "bert": {
        "base":  "bert-base-uncased",
        "large": "bert-large-uncased",
        "type":  "bert",
    },
    "gpt2": {
        "base":  "openai-community/gpt2",
        "large": "openai-community/gpt2-medium",
        "type":  "gpt2",
    },
}

# Default/incumbent hyperparameters the greedy search starts from, and which
# remain in effect for any hyperparameter not yet touched by a search step.
BASE = {"lr": 5e-5, "dropout": 0.1}

# Ordered greedy search steps: (config key to update, label used in experiment
# names, list of candidate values to try). Step order matters: dropout's
# candidates are evaluated using the learning rate already chosen by the
# previous step, not searched jointly/independently.
SEARCH_STEPS = [
    ("lr",      "lr",      [1e-4, 5e-5, 2e-5]),
    ("dropout", "dropout", [0.1, 0.3]),
]


def parse_args():
    """Parses CLI arguments controlling which model families to run, the
    number of seeds per experiment (--runs), training budget (epochs,
    early-stopping patience), batch sizes, and optional convenience flags for
    unattended runs on a remote VM (--notify pings an ntfy.sh topic when done
    or on failure; --shutdown powers off the machine afterwards)."""
    p = argparse.ArgumentParser(
        description="Fine-tuning BERT/GPT-2 (base+large) per intent+slot (Part 2.B).")
    p.add_argument("--models",   nargs="+", default=["bert", "gpt2"], choices=["bert", "gpt2"])
    p.add_argument("--runs",     type=int, default=3)
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--bs_train", type=int, default=32)
    p.add_argument("--bs_eval",  type=int, default=64)
    p.add_argument("--no_save",  action="store_true")
    p.add_argument("--notify",   type=str, default=None)
    p.add_argument("--shutdown", action="store_true")
    return p.parse_args()


def set_seed(seed=42):
    """Seeds Python's random module and PyTorch (CPU + all CUDA devices) for
    reproducibility of the single "save best model" run in save_best()."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Picks the best available compute device: CUDA, then Apple MPS, then
    CPU as the final fallback."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def send_notification(target, best=None, error=None):
    """Posts a push notification via ntfy.sh (or a custom URL) summarizing
    either a fatal error or the best overall result, so a long unattended run
    on a remote machine can be monitored without watching the terminal."""
    url = target if target.startswith("http") else f"https://ntfy.sh/{target}"
    if error is not None:
        title, message = "Run 2.B FALLITA", f"Errore: {type(error).__name__}: {error}"
    elif best is not None:
        title   = f"Run 2.B finita - test F1 {best['slot_f1_mean']:.3f}"
        message = (f"Best: {best['experiment']} ({best['model_name']})\n"
                   f"dev F1 {best['dev_f1_mean']:.3f} | test F1 {best['slot_f1_mean']:.3f} "
                   f"| test acc {best['intent_acc_mean']:.3f}")
    else:
        title, message = "Run 2.B finita", "Nessun risultato registrato."
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"), method="POST")
        req.add_header("Title", title)
        urllib.request.urlopen(req, timeout=10)
        print(f"[notify] notifica inviata a {url}")
    except Exception as e:
        print(f"[notify] invio fallito ({url}): {e}")


def shutdown_machine():
    """Attempts to power off the host machine via a few common shutdown
    commands (with/without sudo), used after unattended runs on a cloud VM to
    avoid leaving (and paying for) an idle instance. Silently gives up if none
    of the commands succeed (e.g. no passwordless sudo)."""
    print("\n[shutdown] spegnimento della macchina richiesto...")
    for cmd in (["sudo", "shutdown", "-h", "now"], ["shutdown", "-h", "now"],
                ["sudo", "systemctl", "poweroff"], ["systemctl", "poweroff"]):
        try:
            subprocess.run(cmd, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("[shutdown] impossibile spegnere (serve sudo senza password).")


def build_loaders(model_type, lang, train_raw, dev_raw, test_raw, args):
    """Costruisce i DataLoader per il tipo di modello (bert o gpt2).
    Il tokenizer base e' valido anche per large/medium (stesso vocabolario).

    English: Builds the train/dev/test DataLoaders for the given model_type,
    selecting the matching Dataset class, tokenizer, and collate_fn (see
    utils.py). The tokenizer is always loaded from the BASE model id of the
    family (MODEL_FAMILIES[...]["base"]) because base and large/medium share
    the same vocabulary — these loaders are therefore reused unchanged when
    later fine-tuning the large/medium variant of the same family.
    """
    if model_type == "bert":
        tokenizer = get_bert_tokenizer(MODEL_FAMILIES["bert"]["base"])
        ds        = lambda raw: BERTIntentsAndSlots(raw, lang, tokenizer)
        collate   = collate_fn_bert
    else:
        tokenizer = get_gpt2_tokenizer(MODEL_FAMILIES["gpt2"]["base"])
        ds        = lambda raw: GPT2IntentsAndSlots(raw, lang, tokenizer)
        collate   = collate_fn_gpt2

    return get_dataloaders(ds(train_raw), ds(dev_raw), ds(test_raw), collate,
                           batch_size_train=args.bs_train, batch_size_eval=args.bs_eval)


def main():
    """CLI entry point: runs the full search (run_search), and regardless of
    success or failure optionally sends a notification and/or shuts down the
    machine (useful for unattended cloud runs); re-raises any exception after
    those finally-clauses so failures are still visible/non-silent."""
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
    """Orchestrates the full Part 2.B experiment grid across model families.

    For each family in args.models (bert, gpt2):
      1. Build the shared train/dev/test DataLoaders once (valid for both the
         base and large/medium variant of the family).
      2. Run the 2-step greedy search (lr, then dropout) on the BASE model
         only, each candidate evaluated over args.runs seeds via
         run_experiments, propagating the winning value of each step into
         `config` before the next step starts (so dropout's candidates are
         tried with the already-chosen winning lr).
      3. Fine-tune the family's LARGE/MEDIUM variant exactly once, reusing the
         winning base config — deliberately not re-searched, since searching
         lr/dropout again on the bigger model would be much more expensive
         for a comparison that (as the results show) doesn't pay off.
      4. Pick the family's best of {base, large/medium} by dev Slot F1, and
         optionally save its weights via save_best().
    Every experiment is looked up against already-logged results
    (done_experiments/load_results) before running and skipped if found,
    making the whole search resumable/idempotent across interrupted runs.
    Returns the single best record (by dev Slot F1) across all families.
    """
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(BIN_DIR, exist_ok=True)

    tmp_train_raw = load_data(TRAIN_PATH)
    test_raw      = load_data(TEST_PATH)
    train_raw, dev_raw = create_dev_split(tmp_train_raw, dev_size=0.10)
    lang = build_lang(train_raw, dev_raw, test_raw)
    slots_size, n_intents = len(lang.slot2id), len(lang.intent2id)
    print(f"Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}")
    print(f"Slot: {slots_size} | Intent: {n_intents}\n")

    done    = done_experiments(RESULTS_JSON)
    records = {r["experiment"]: r for r in load_results(RESULTS_JSON)}
    global_best = None

    for family in args.models:
        fam        = MODEL_FAMILIES[family]
        model_type = fam["type"]
        base_name  = fam["base"]
        large_name = fam["large"]
        large_key  = "bert-large" if family == "bert" else "gpt2-medium"

        print(f"\n########## FAMIGLIA: {family} ##########")
        print(f"  base  : {base_name}")
        print(f"  large : {large_name}")

        # Built once per family and reused for both the base greedy search
        # and the large/medium run, since tokenization/vocabulary is shared
        # within a family (see build_loaders docstring).
        loaders       = build_loaders(model_type, lang, train_raw, dev_raw, test_raw, args)
        make_datasets = lambda: loaders

        def get_or_run(name, param, cfg, mn):
            """Idempotent skip-if-done runner: returns the cached record for
            `name` if already present in results.json, otherwise runs
            run_experiments for that exact (param, cfg, model_name)
            combination, logs the new record, regenerates the Markdown
            report, and marks `name` as done."""
            if name in done:
                print(f"[skip] {name} (gia' in {RESULTS_JSON})")
                return records[name]
            info = run_experiments(
                make_datasets, lang, slots_size, n_intents,
                lr=cfg["lr"], model_name=mn, model_type=model_type,
                dropout=cfg["dropout"], n_runs=args.runs, n_epochs=args.epochs,
                patience=args.patience, experiment_name=name, seed=args.seed, device=device,
            )
            rec = make_record(name, param, info, seed=args.seed)
            append_result(RESULTS_JSON, rec)
            regenerate_observations_md(RESULTS_JSON, OBSERVATIONS_MD)
            done.add(name)
            records[name] = rec
            return rec

        # --- Ricerca greedy sul modello BASE ---
        # `config` starts at BASE defaults and is mutated step by step: each
        # step's winning value is written into config[param] before the next
        # step runs, so later steps (and the eventual large/medium run) always
        # see the already-decided winners of earlier steps.
        config = dict(BASE)
        best_score, model_best = None, None
        for param, label, candidates in SEARCH_STEPS:
            print(f"\n--- STEP '{param}' ({family}-base, incumbent = {config[param]}) ---")
            step_value, step_score, step_rec = config[param], best_score, model_best
            for v in candidates:
                name = f"{family}_{label}{v}"
                rec  = get_or_run(name, param, {**config, param: v}, base_name)
                score = rec["dev_f1_mean"]
                print(f"  {name}: dev F1 = {score:.4f} (test F1 {rec['slot_f1_mean']:.4f})")
                # Track the best-by-dev-F1 candidate seen in this step; ties
                # keep the first (earliest-tried) candidate since only a
                # strict improvement (score > step_score) replaces it.
                if step_score is None or score > step_score:
                    step_value, step_score, step_rec = v, score, rec
            config[param] = step_value
            best_score, model_best = step_score, step_rec
            print(f"  -> '{param}' = {step_value} (dev F1 = {step_score:.4f})")

        print(f"\n>>> Migliore {family}-base: {model_best['experiment']} "
              f"| dev F1 {model_best['dev_f1_mean']:.4f} "
              f"| test F1 {model_best['slot_f1_mean']:.4f} "
              f"| test acc {model_best['intent_acc_mean']:.4f}")

        # --- Esperimento sul modello LARGE/MEDIUM con la config migliore ---
        # Reuses the winning (lr, dropout) from the base greedy search above
        # WITHOUT repeating the search on the larger model — a single
        # fine-tuning run (over args.runs seeds) purely to check whether
        # scaling up the backbone helps, not a new independent search.
        large_exp_name = f"{large_key}_lr{config['lr']}_dropout{config['dropout']}"
        print(f"\n--- LARGE/MEDIUM: {large_name} (config: lr={config['lr']}, dropout={config['dropout']}) ---")
        rec_large = get_or_run(large_exp_name, "large", config, large_name)
        print(f"  {large_exp_name}: dev F1 = {rec_large['dev_f1_mean']:.4f} "
              f"(test F1 {rec_large['slot_f1_mean']:.4f})")

        # Migliore tra base e large per questa famiglia (selezione sulla dev F1)
        family_best    = max([model_best, rec_large], key=lambda r: r["dev_f1_mean"])
        family_best_mn = large_name if family_best is rec_large else base_name
        print(f"\n>>> Migliore {family} (base vs large): {family_best['experiment']} "
              f"| dev F1 {family_best['dev_f1_mean']:.4f} "
              f"| test F1 {family_best['slot_f1_mean']:.4f} "
              f"| test acc {family_best['intent_acc_mean']:.4f}")

        if not args.no_save:
            save_best(family_best_mn, model_type, config, family_best, loaders,
                      lang, slots_size, n_intents, device, args)

        if global_best is None or family_best["dev_f1_mean"] > global_best["dev_f1_mean"]:
            global_best = family_best

    print("\n" + "=" * 70)
    print(f"MIGLIORE GLOBALE (dev F1): {global_best['experiment']} ({global_best['model_name']})")
    print(f"  test Slot F1 {global_best['slot_f1_mean']:.4f} | "
          f"test Intent Acc {global_best['intent_acc_mean']:.4f}")
    print("=" * 70)
    print(f"\nRisultati JSON: {RESULTS_JSON}")
    print(f"Report osservazioni: {OBSERVATIONS_MD}")
    return global_best


def save_best(model_name, model_type, config, model_best, loaders,
              lang, slots_size, n_intents, device, args):
    """Allena UNA istanza della config migliore e salva state_dict + vocabolari.

    English: Re-trains a single instance (1 run, deterministic seed) of the
    winning configuration found above (identified by model_best['experiment'])
    purely to obtain a persisted checkpoint, since run_experiments/
    run_search keep only metrics, not model weights, for the multi-seed
    search runs. Skips re-training if a checkpoint for this experiment
    already exists in BIN_DIR (idempotent, like the rest of the search).
    Saves the state_dict together with model_name, model_type, the
    hyperparameter config, and the slot/intent vocabularies needed to
    reload and use the model later for inference.
    """
    best_pt = os.path.join(BIN_DIR, f"{model_best['experiment']}.pt")
    if os.path.exists(best_pt):
        print(f"[bin] Modello migliore gia' presente: {best_pt}")
        return

    print(f"[bin] Alleno '{model_best['experiment']}' (1 run) per salvarlo in {best_pt} ...")
    set_seed(args.seed)
    train_loader, dev_loader, _ = loaders
    if model_type == "bert":
        model = BERTforNLU(slots_size, n_intents, model_name, config["dropout"]).to(device)
    else:
        model = GPT2forNLU(slots_size, n_intents, model_name, config["dropout"]).to(device)

    optimizer         = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion_slots   = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_intents = nn.CrossEntropyLoss()
    best_model, best_f1, best_acc = train_model(
        model, train_loader, dev_loader, lang, optimizer,
        criterion_slots, criterion_intents, model_type=model_type,
        n_epochs=args.epochs, patience=args.patience,
        experiment_name=f"{model_best['experiment']}_BEST",
    )
    torch.save({"model": best_model.state_dict(), "model_name": model_name,
                "model_type": model_type, "config": config,
                "slot2id": lang.slot2id, "intent2id": lang.intent2id}, best_pt)
    print(f"  Modello salvato in: {best_pt} (dev F1={best_f1:.4f}, dev Acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
