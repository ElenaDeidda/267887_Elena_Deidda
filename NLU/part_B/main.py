# main.py
# Part 2.B - Fine-tuning di BERT (base+large) e GPT-2 (base+medium) per intent + slot su ATIS.
#
# Per OGNI famiglia (bert, gpt2):
#   Step 0 - ricerca greedy del learning rate sul modello BASE
#   Step 1 - ricerca greedy del dropout sul modello BASE (con il miglior lr)
#   Step 2 - esperimento singolo sul modello LARGE/MEDIUM con la config migliore
#
# bert-base e bert-large condividono lo stesso tokenizer WordPiece (stesso vocab),
# idem gpt2 e gpt2-medium (BPE): i DataLoader costruiti per il base valgono anche
# per la variante large/medium.
#
# Selezione sulla DEV F1; ogni configurazione e' ripetuta su piu' run
# (media +- std di Slot F1 conll e Intent Accuracy).
# Risultati in results/results.json e observations.md. Modelli migliori in bin/.
#
# Uso:
#   python main.py
#   python main.py --models bert gpt2 --runs 3 --epochs 30
#   python main.py --notify <topic_ntfy> --shutdown   # comodo sulla VM

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

BASE = {"lr": 5e-5, "dropout": 0.1}

SEARCH_STEPS = [
    ("lr",      "lr",      [1e-4, 5e-5, 2e-5]),
    ("dropout", "dropout", [0.1, 0.3]),
]


def parse_args():
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
    Il tokenizer base e' valido anche per large/medium (stesso vocabolario)."""
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

        loaders       = build_loaders(model_type, lang, train_raw, dev_raw, test_raw, args)
        make_datasets = lambda: loaders

        def get_or_run(name, param, cfg, mn):
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
    """Allena UNA istanza della config migliore e salva state_dict + vocabolari."""
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
