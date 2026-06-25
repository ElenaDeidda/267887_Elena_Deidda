# functions.py
# Part 1.B - Freeze del backbone + abilitazione LoRA, loop di training/eval,
# early stopping e logging degli esperimenti in JSON (come la Part 1.A).
#
# A differenza della Part 1.A, la loss e' calcolata dentro GPT2LMHeadModel:
# passando `labels` al forward, il modello restituisce output.loss (fa lo shift
# internamente). Il padding e' gestito mettendo le posizioni di pad a -100
# (HuggingFace ignora -100). Solo le matrici LoRA sono addestrabili.

import os
import json
import math
import copy
from datetime import datetime

import torch
from tqdm import tqdm

from model import CustomGPT2Attention

# La consegna chiede PPL test < 250 (e migliore della Part 1.A).
PPL_THRESHOLD = 250


# ----------------------------------------------------------------------------
# 1. Freeze del backbone / abilitazione dei soli adapter LoRA
# ----------------------------------------------------------------------------

def freeze_pretrained_and_enable_lora(model):
    """Congela tutti i parametri e riabilita solo le matrici LoRA di ogni
    CustomGPT2Attention (il backbone da ~124M resta congelato)."""
    for param in model.parameters():
        param.requires_grad = False

    lora_attrs = ["lora_A_q", "lora_A_k", "lora_A_v",
                  "lora_B_q", "lora_B_k", "lora_B_v"]

    for module in model.modules():
        if isinstance(module, CustomGPT2Attention):
            for attr_name in lora_attrs:
                for param in getattr(module, attr_name).parameters():
                    param.requires_grad = True


# ----------------------------------------------------------------------------
# 2. Loop di training (una epoca)
# ----------------------------------------------------------------------------

def train_loop(data, optimizer, model, tokenizer, clip=5.0):
    """Una epoca di training. Restituisce la cross-entropy media per token (pesata)."""
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="  train", leave=False)
    for input_ids, _labels, n_tokens in pbar:
        optimizer.zero_grad()

        # labels = copia di input_ids; pad -> -100 (ignorato da HuggingFace nella loss).
        # Lo shift dei label lo fa il modello internamente.
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100

        output = model(input_ids, labels=labels)  # logits + loss interni

        # output.loss e' la media per token: *n_tokens per la loss totale del batch
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)

        output.loss.backward()
        # gradient clipping: i gradienti fluiscono dal backbone congelato alle matrici LoRA
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)


# ----------------------------------------------------------------------------
# 3. Loop di valutazione
# ----------------------------------------------------------------------------

def eval_loop(data, model, tokenizer):
    """Valuta su dev/test senza aggiornare i pesi. Restituisce (ppl, loss_per_token)."""
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, _labels, n_tokens in tqdm(data, desc="  eval", leave=False):
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100

            output = model(input_ids, labels=labels)

            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_avg = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(min(loss_avg, 100))  # cap a e^100 per evitare overflow nelle prime epoche
    return ppl, loss_avg


# ----------------------------------------------------------------------------
# 4. Training completo con early stopping (sulla dev PPL)
# ----------------------------------------------------------------------------

def train_model(model, train_loader, dev_loader, tokenizer, optimizer,
                n_epochs=20, patience=3, experiment_name="exp"):
    """Training con early stopping sulla dev PPL. Col modello pre-addestrato la
    convergenza e' rapida (poche epoche).

    Returns:
        best_model (su CPU), best_dev_ppl, best_epoch, epochs_run
    """
    best_ppl = math.inf
    best_model = None
    best_epoch = -1
    epochs_run = 0
    cur_patience = patience

    pbar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")
    for epoch in pbar:
        epochs_run = epoch
        train_loss = train_loop(train_loader, optimizer, model, tokenizer)
        dev_ppl, _ = eval_loop(dev_loader, model, tokenizer)
        pbar.set_postfix(loss=f"{train_loss:.4f}", dev_ppl=f"{dev_ppl:.2f}")

        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            best_epoch = epoch
            best_model = copy.deepcopy(model).cpu()
            cur_patience = patience
        else:
            cur_patience -= 1
            if cur_patience <= 0:
                break

    return best_model, best_ppl, best_epoch, epochs_run


def count_trainable(model):
    """Numero di parametri addestrabili (le sole matrici LoRA dopo il freeze)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# 5. Logging degli esperimenti (JSON) e report Markdown - come la Part 1.A
# ----------------------------------------------------------------------------

def load_results(path):
    """Carica la lista dei run gia' salvati (lista vuota se il file non esiste)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def done_experiments(path):
    """Insieme dei nomi di esperimento gia' presenti nel JSON (per l'idempotenza)."""
    return {r["experiment"] for r in load_results(path)}


def append_result(path, record):
    """Aggiunge un record di esperimento al file JSON (creandolo se serve)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results = load_results(path)
    results.append(record)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def make_record(experiment, step, info, seed=42):
    """Costruisce il dizionario da salvare nel JSON a partire dalle metriche di run."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "step": step,
        "rank": info["rank"],
        "alpha": info["alpha"],
        "scaling": round(info["alpha"] / info["rank"], 4),
        "lr": info["lr"],
        "batch_size": info["batch_size"],
        "n_trainable": info["n_trainable"],
        "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"],
        "best_dev_ppl": round(float(info["best_dev_ppl"]), 4),
        "test_ppl": round(float(info["test_ppl"]), 4),
        "test_loss": round(float(info["test_loss"]), 4),
        "passes_threshold": bool(info["test_ppl"] < PPL_THRESHOLD),
        "seed": seed,
    }


_STEP_DESC = {
    0: "Step 0 - Ricerca del learning rate (rank/alpha fissi).",
    1: "Step 1 - Rango r di LoRA (alpha = 2*r, scaling = 2.0).",
    2: "Step 2 - Alpha (a parita' del miglior rank; scaling = alpha/rank).",
}


def regenerate_observations_md(results_path, md_path, baseline_ppl=None):
    """Rigenera il report Markdown a partire dal JSON dei risultati.

    La selezione del migliore (per step e globale) si fa SEMPRE sulla dev PPL;
    il test PPL e' solo riportato come numero finale.
    """
    results = load_results(results_path)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    lines = ["# Osservazioni esperimenti - LM Part 1.B (LoRA)", ""]
    lines.append(f"_Generato automaticamente da main.py. "
                 f"Ultimo aggiornamento: {datetime.now().isoformat(timespec='seconds')}._")
    lines.append("")
    target = f"PPL test < {PPL_THRESHOLD}"
    if baseline_ppl is not None:
        target += f" e migliore della Part 1.A (test PPL {baseline_ppl})"
    lines.append(f"Vincolo della consegna: **{target}**.")
    lines.append("")

    if not results:
        lines.append("_Nessun esperimento registrato finora._")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    best = min(results, key=lambda r: r["best_dev_ppl"])
    lines.append("## Migliore configurazione finora")
    lines.append("")
    lines.append(f"- selezione su **best dev PPL: {best['best_dev_ppl']:.2f}** "
                 f"-> **Test PPL: {best['test_ppl']:.2f}** "
                 f"({'OK <250' if best['passes_threshold'] else 'NON soddisfa <250'})")
    lines.append(f"- esperimento: `{best['experiment']}`")
    lines.append(f"- rank: `{best['rank']}` | alpha: `{best['alpha']}` | "
                 f"scaling: `{best['scaling']}` | lr: `{best['lr']}`")
    lines.append(f"- parametri addestrabili (LoRA): {best['n_trainable']:,}")
    lines.append("")

    # raggruppa per step
    steps = {}
    for r in results:
        steps.setdefault(r["step"], []).append(r)

    for s in sorted(steps.keys(), key=lambda x: (x is None, x)):
        runs = steps[s]
        lines.append(f"## Step {s}")
        desc = _STEP_DESC.get(s)
        if desc:
            lines.append(f"_{desc}_")
        lines.append("")
        lines.append("| esperimento | rank | alpha | scaling | lr | epoche | dev PPL | test PPL | <250 | |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        best_run = min(runs, key=lambda r: r["best_dev_ppl"])
        for r in runs:
            star = "*" if r is best_run else ""
            ok = "si" if r["passes_threshold"] else "**no**"
            lines.append(
                f"| {r['experiment']} | {r['rank']} | {r['alpha']} | {r['scaling']} "
                f"| {r['lr']} | {r['epochs_run']} | {r['best_dev_ppl']:.2f} "
                f"| {r['test_ppl']:.2f} | {ok} | {star} |"
            )
        lines.append("")
        lines.append(f"- Osservazione (scelta sul dev): migliore `{best_run['experiment']}` "
                     f"(dev PPL {best_run['best_dev_ppl']:.2f}, test PPL {best_run['test_ppl']:.2f}).")
        lines.append("- Note (da completare nel report): ")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
