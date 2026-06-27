# main.py
# Punto di ingresso per gli esperimenti incrementali della Part 1.A.
#
# FILOSOFIA D'USO (allineata alla consegna):
#  - Per capire QUALE iperparametro conviene, lanci UN GRUPPO alla volta:
#        python main.py d_model
#    Dentro al gruppo i diversi valori (es. 128 / 256 / 384) vengono provati uno
#    dopo l'altro SENZA fermarsi: ti restituisce la PPL di ciascuno cosi' puoi
#    scegliere il migliore.
#  - Costruisci la config in modo INCREMENTALE bloccando i valori gia' scelti via
#    override da riga di comando, senza editare il file. Esempio:
#        python main.py d_model                      # scegli, poniamo, 384
#        python main.py n_heads     --d_model 384    # tieni 384, vari le teste
#        python main.py num_layers  --d_model 384 --n_heads 6
#        python main.py ff_dim      --d_model 384 --n_heads 6 --num_layers 6
#        python main.py dropout     --d_model 384 --n_heads 6 --num_layers 6 --ff_dim 1536
#        python main.py weight_tying --d_model 384 --n_heads 6 --num_layers 6 --ff_dim 1536 --dropout 0.1
#  - Per provare lo stesso iperparametro con valori tuoi:
#        python main.py d_model --values 256,512,768
#  - Per un singolo esperimento ad-hoc (config finale, prova mirata):
#        python main.py single --d_model 384 --n_heads 6 --num_layers 6 --ff_dim 1536 \
#                              --dropout 0.1 --weight_tying --lr 3e-4
#
# Ogni run viene salvato in results/results.json e il report leggibile
# results/observations.md viene rigenerato automaticamente.

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
# Configurazione di base (il "punto di partenza" del modello baseline).
# Gli override da riga di comando (--d_model, --lr, ...) la modificano prima
# di lanciare lo sweep, permettendo la costruzione incrementale.
# ----------------------------------------------------------------------------
BASE_CONFIG = dict(
    pos_emb_size=1024,
    d_model=256,
    n_heads=4,
    num_layers=4,
    ff_dim=1024,
    dropout=0.0,
    weight_tying=False,
)
BASE_LR = 5e-4

# ----------------------------------------------------------------------------
# Definizione degli sweep: gruppo -> (chiave variata, lista di valori di default).
# 'lr' e' trattato a parte perche' non e' un iperparametro del modello.
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
# Parsing degli argomenti
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Esperimenti incrementali GPT-2 (LM Part 1.A).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("group", choices=GROUPS,
                   help="quale gruppo di esperimenti lanciare")

    # Override degli iperparametri di base (per la costruzione incrementale).
    p.add_argument("--pos_emb_size", type=int)
    p.add_argument("--d_model", type=int)
    p.add_argument("--n_heads", type=int)
    p.add_argument("--num_layers", type=int)
    p.add_argument("--ff_dim", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--weight_tying", action="store_true", default=None,
                   help="attiva il weight tying nella config di base")
    p.add_argument("--lr", type=float, help="learning rate (override di BASE_LR)")

    # Valori personalizzati per lo sweep (es. --values 256,512,768)
    p.add_argument("--values", type=str,
                   help="lista di valori per lo sweep, separati da virgola")

    # Parametri di training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=3)
    # cambiata da default 8 per migliorare le prestazioni
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_best", action="store_true",
                   help="salva lo state_dict del modello migliore di questa run in bin/")
    p.add_argument("--shutdown", action="store_true",
                   help="spegne la macchina al termine (anche in caso di errore)")
    p.add_argument("--notify", type=str, default=None,
                   help="topic ntfy.sh (es. 'elena-nlu-vm-x7k2') o URL/webhook "
                        "completo a cui mandare una notifica a fine run")
    return p.parse_args()


def send_notification(target, best_record=None, error=None):
    """Manda una notifica push a fine run (ntfy.sh o qualsiasi webhook HTTP POST).

    'target' puo' essere un semplice topic ntfy (es. 'elena-nlu-vm-x7k2') oppure
    un URL completo. Il messaggio riassume la migliore PPL o l'eventuale errore.
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
    """Spegne la macchina virtuale. Prova diversi comandi per robustezza."""
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
    """Applica gli override CLI alla config e al learning rate di base."""
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
    """Converte la stringa --values nel tipo corretto in base alla chiave variata."""
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if key in ("lr", "dropout"):
        return [float(s) for s in parts]
    if key == "weight_tying":
        return [s.lower() in ("1", "true", "yes", "si", "sì") for s in parts]
    return [int(s) for s in parts]


def run_sweep(group, key, values, base_cfg, base_lr, loaders, tokenizer, device, args):
    """Esegue tutti i valori di un gruppo, uno dopo l'altro, e li registra.

    Returns:
        (best_model, best_record) del valore con test PPL piu' bassa nel gruppo.
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

        # salta combinazioni invalide (d_model non divisibile per n_heads)
        if cfg["d_model"] % cfg["n_heads"] != 0:
            print(f"[skip] d_model={cfg['d_model']} non divisibile per "
                  f"n_heads={cfg['n_heads']}: combinazione saltata.")
            continue

        label = f"{group}={val}"
        set_seed(args.seed)  # stesso seed per ogni run -> confronto equo
        model, info = run_experiment(
            label, cfg, loaders, tokenizer, device,
            lr=lr, n_epochs=args.epochs, patience=args.patience,
        )

        record = make_record(group, label, info,
                             swept_key=key, swept_value=val, seed=args.seed)
        append_result(RESULTS_JSON, record)
        regenerate_observations_md(RESULTS_JSON, OBSERVATIONS_MD)

        # selezione del migliore SEMPRE sul dev set (mai sul test): info["best_dev_ppl"]
        if best_record is None or info["best_dev_ppl"] < best_record["best_dev_ppl"]:
            best_model, best_record = model, record

    if best_record is not None:
        print(f"\n>>> Migliore del gruppo '{group}' (scelto su dev): {best_record['label']} "
              f"-> dev PPL {best_record['best_dev_ppl']:.2f} | test PPL {best_record['test_ppl']:.2f}")
    return best_model, best_record


def main():
    args = parse_args()
    best_record = None
    err = None
    try:
        best_record = _run(args)
    except Exception as e:
        err = e
    finally:
        # Ordine corretto: training (gia' fatto) -> notifica -> shutdown.
        # Eseguito sempre, cosi' avvisa/spegne anche se il training va in errore.
        if args.notify:
            send_notification(args.notify, best_record=best_record, error=err)
        if args.shutdown:
            shutdown_machine()
    # Ri-solleva l'eventuale errore DOPO aver notificato/spento, per avere
    # comunque un exit code corretto e il traceback nei log.
    if err is not None:
        raise err


def _run(args):
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
        # Un singolo esperimento con la config di base (eventuali override).
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
        # Comodita': esegue tutti gli sweep di fila sulla STESSA config di base.
        # NB: non e' la selezione incrementale (quella la fai tu un gruppo alla
        # volta), ma una panoramica completa.
        for g, (key, default_values) in SWEEPS.items():
            values = parse_values(args.values, key) if args.values else default_values
            m, r = run_sweep(g, key, values, base_cfg, base_lr,
                             loaders, tokenizer, device, args)
            if r is not None and (best_record is None or r["best_dev_ppl"] < best_record["best_dev_ppl"]):
                best_model, best_record = m, r

    else:
        # Un gruppo di sweep singolo (il caso d'uso principale).
        key, default_values = SWEEPS[args.group]
        values = parse_values(args.values, key) if args.values else default_values
        best_model, best_record = run_sweep(
            args.group, key, values, base_cfg, base_lr,
            loaders, tokenizer, device, args,
        )

    # Salvataggio opzionale del miglior modello di questa esecuzione.
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
