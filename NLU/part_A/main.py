# main.py
# Part 2.A - NLU from scratch (Intent Classification + Slot Filling) su ATIS.
#
# Esegue la ricerca incrementale (greedy) degli iperparametri del GPT-2 from
# scratch. Ogni configurazione e' ripetuta su piu' run (media +- std di Slot F1
# conll e Intent Accuracy).
#
# RICERCA GREEDY (un iperparametro alla volta, vincitore scelto sulla DEV F1):
#   Step 0 - learning rate
#   Step 1 - architettura: d_model, n_heads, num_layers, ff_dim
#   Step 2 - dropout (incluso quello prima delle teste di output)
#
# Il vincitore di ogni step e' propagato automaticamente allo step successivo.
# I risultati vanno in results/results.json (idempotente) e observations.md.
#
# Uso:
#   python main.py
#   python main.py --runs 5 --epochs 200
#   python main.py --notify <topic_ntfy> --shutdown   # comodo sulla VM

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
# Percorsi
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join("dataset", "ATIS")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")

RESULTS_DIR = "results"
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
OBSERVATIONS_MD = os.path.join(RESULTS_DIR, "observations.md")
BIN_DIR = "bin"

# ----------------------------------------------------------------------------
# Config di partenza e ricerca greedy (un iperparametro alla volta)
# ----------------------------------------------------------------------------
BASE = {"lr": 0.01, "d_model": 64, "n_heads": 2,
        "num_layers": 2, "ff_dim": 256, "dropout": 0.0}

# (param variato, etichetta, lista di candidati). L'incumbent (valore corrente)
# non e' rieseguito: il suo punteggio e' gia' noto e viene confrontato.
SEARCH_STEPS = [
    ("lr",         "step0_lr",     [0.1, 0.01, 0.001, 0.0001]),
    ("d_model",    "step1_dmodel", [128, 256]),
    ("n_heads",    "step1_nheads", [4, 8]),
    ("num_layers", "step1_layers", [4, 6]),
    ("ff_dim",     "step1_ffdim",  [512, 1024]),
    ("dropout",    "step2_dropout", [0.1, 0.2, 0.3]),
]


def parse_args():
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
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def send_notification(target, best=None, error=None):
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
    """Carica ATIS, crea il dev set, costruisce i vocabolari e i DataLoader."""
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
    """Allena UNA istanza della config migliore e salva state_dict + vocabolari in bin/."""
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
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(BIN_DIR, exist_ok=True)

    loaders, lang, dims = setup_data(device)
    train_loader, dev_loader, test_loader = loaders
    vocab_len, slots_len, n_intents = dims

    done = done_experiments(RESULTS_JSON)
    records = {r["experiment"]: r for r in load_results(RESULTS_JSON)}

    def get_or_run(name, param, cfg):
        """Record di `cfg`: dal JSON se gia' presente, altrimenti addestra (runs) e salva."""
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

    # cascata greedy: a ogni step si tiene il vincitore sulla DEV F1
    config = dict(BASE)
    best_score = None   # dev F1 dell'incumbent
    best_record = None

    for param, label, candidates in SEARCH_STEPS:
        print(f"\n########## STEP '{param}' (incumbent = {config[param]}) ##########")
        step_value, step_score, step_rec = config[param], best_score, best_record
        for v in candidates:
            name = f"{label}{v}"
            rec = get_or_run(name, param, {**config, param: v})
            score = rec["dev_f1_mean"]
            print(f"  {name}: dev F1 = {score:.4f} (test F1 {rec['slot_f1_mean']:.4f})")
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
