# main.py
# Part 1.B - Fine-tuning di GPT-2 pre-addestrato con LoRA sul Penn TreeBank.
#
# Vengono addestrate solo le matrici LoRA su query/key/value (~0.2-1% dei
# parametri); i ~124M pesi originali restano congelati.
#
# RICERCA GREEDY (un gruppo alla volta, vincitore scelto sulla DEV PPL):
#   Step 0 - learning rate          (rank=4, alpha=8 fissi)
#   Step 1 - rank r in {4, 8, 16}   (alpha = 2*r -> scaling 2.0)
#   Step 2 - alpha (al miglior rank; scaling = alpha/rank)
#
# Il vincitore di ogni step e' propagato automaticamente allo step successivo.
# Ogni run viene salvato in results/results.json (idempotente: gli esperimenti
# gia' presenti vengono saltati) e results/observations.md viene rigenerato.
#
# Uso:
#   python main.py                         # esegue tutta la ricerca greedy
#   python main.py --epochs 20 --seed 42
#   python main.py --notify <topic_ntfy> --shutdown   # comodo sulla VM
#
# Target: PPL test < 250 (obbligatorio) e migliore della Part 1.A.

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
# Percorsi
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
# Config di partenza e ricerca greedy
# ----------------------------------------------------------------------------
BASE = {"rank": 4, "alpha": 8, "lr": 5e-4, "batch_size": 8}

SEARCH_STEPS = [
    # Step 0 - learning rate (rank=4, alpha=8 fissi)
    (0, [
        ("step0_lr1e-3", {"lr": 1e-3}),
        ("step0_lr5e-4", {"lr": 5e-4}),
        ("step0_lr1e-4", {"lr": 1e-4}),
    ]),
    # Step 1 - rank (alpha = 2*rank -> scaling 2.0)
    (1, [
        ("step1_rank4",  {"rank": 4,  "alpha": 8}),
        ("step1_rank8",  {"rank": 8,  "alpha": 16}),
        ("step1_rank16", {"rank": 16, "alpha": 32}),
    ]),
    # Step 2 - alpha al miglior rank (scaling = alpha/rank)
    (2, [
        ("step2_alpha_half", {"alpha_factor": 0.5}),   # alpha = rank/2
        ("step2_alpha_eq",   {"alpha_factor": 1.0}),   # alpha = rank
        ("step2_alpha_2x",   {"alpha_factor": 2.0}),   # alpha = 2*rank
    ]),
]


def parse_args():
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
    """Notifica push a fine run (ntfy.sh o qualsiasi webhook HTTP POST)."""
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

    tokenizer = get_tokenizer()
    train_raw = read_file(TRAIN_PATH)
    dev_raw = read_file(DEV_PATH)
    test_raw = read_file(TEST_PATH)
    print(f"Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}")
    print(f"Target: PPL test < {PPL_THRESHOLD} e migliore della Part 1.A\n")

    done = done_experiments(RESULTS_JSON)
    records = {r["experiment"]: r for r in load_results(RESULTS_JSON)}

    def get_or_run(name, step, cfg):
        """Restituisce il record di `cfg`: dal JSON se gia' presente, altrimenti
        addestra (1 run) e lo salva. Idempotente."""
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

    # cascata greedy: a ogni step si tiene il vincitore sulla DEV PPL
    config = dict(BASE)
    best_record = None

    for step, candidates in SEARCH_STEPS:
        print(f"\n########## STEP {step} ##########")
        step_best = None
        for name, override in candidates:
            trial = dict(config)
            # lo step 2 esprime alpha come fattore del rank corrente
            if "alpha_factor" in override:
                trial["alpha"] = int(round(config["rank"] * override["alpha_factor"]))
            trial.update({k: v for k, v in override.items() if k != "alpha_factor"})

            rec = get_or_run(name, step, trial)
            d = rec["best_dev_ppl"]
            print(f"  {name}: rank={trial['rank']} alpha={trial['alpha']} "
                  f"lr={trial['lr']} -> dev PPL {d:.2f}")
            if step_best is None or d < step_best["best_dev_ppl"]:
                step_best = rec
        # propaga il vincitore dello step alla config corrente
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

    # ri-alleno la config migliore per salvare i pesi LoRA in bin/
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
    """Esegue un singolo esperimento LoRA: GPT-2 pre-addestrato + adapter, freeze,
    training (early stopping su dev PPL), valutazione sul test. Restituisce il record."""
    print(f"\n===== Esperimento: {name} =====")
    print(f"  rank={cfg['rank']} alpha={cfg['alpha']} "
          f"scaling={cfg['alpha']/cfg['rank']:.2f} lr={cfg['lr']} bs={cfg['batch_size']}")
    set_seed(args.seed)

    train_loader, dev_loader, test_loader = get_dataloaders(
        train_raw, dev_raw, test_raw, tokenizer, device,
        batch_size_train=cfg["batch_size"], batch_size_eval=cfg["batch_size"] * 2,
    )

    model = GPT2_LoRA.from_pretrained(
        "openai-community/gpt2", rank=cfg["rank"], alpha=cfg["alpha"],
    )
    model.to(device)
    freeze_pretrained_and_enable_lora(model)
    param_stats(model)

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
        # salva solo i pesi LoRA (molto piu' leggeri del modello completo)
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
