# functions.py
# Part 2.A - Training e valutazione per il joint intent classification + slot filling,
# piu' il logging degli esperimenti in JSON (come la Part 1.A).
#
# Loss congiunta = CE(intent) + CE(slot), con la loss sugli slot che usa
# ignore_index=PAD_TOKEN (ignora sia il padding sia la posizione del CLS). Gli
# slot sono valutati con la F1 a livello di chunk (conll), gli intent con
# l'accuracy. Ogni configurazione viene ripetuta su piu' run (media +- std) e
# l'early stopping segue la slot F1 di dev.

import os
import json
import copy
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from conll import evaluate
from sklearn.metrics import classification_report

from model import GPT2, init_weights, count_parameters

PAD_TOKEN = 0


# ----------------------------------------------------------------------------
# 1. Loop di training (una epoca)
# ----------------------------------------------------------------------------

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """Una epoca di training. Restituisce la lista delle loss per batch."""
    model.train()
    loss_array = []
    for batch in tqdm(data, desc="  train", leave=False):
        optimizer.zero_grad()
        slots, intent = model(batch["utterances"], batch["slots_len"])
        slots = slots.permute(0, 2, 1)  # CrossEntropyLoss vuole (B, C, L)

        loss_intent = criterion_intents(intent, batch["intents"])
        loss_slot = criterion_slots(slots, batch["y_slots"])
        loss = loss_intent + loss_slot  # joint loss a pesi uguali
        loss_array.append(loss.item())

        loss.backward()
        optimizer.step()
    return loss_array


# ----------------------------------------------------------------------------
# 2. Loop di valutazione (dev o test)
# ----------------------------------------------------------------------------

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """Valuta senza aggiornare i pesi: decodifica intent e slot e calcola le metriche.

    La decodifica slot usa length = slots_len[i] - 1 per escludere il CLS, perche'
    conll.evaluate() si aspetta sequenze senza il token CLS.

    Returns:
        results (dict conll, results['total']['f'] = slot F1),
        report_intent (dict, report_intent['accuracy']),
        loss_array
    """
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []

    with torch.no_grad():
        for batch in tqdm(data, desc="  eval", leave=False):
            slots, intent = model(batch["utterances"], batch["slots_len"])
            slots = slots.permute(0, 2, 1)  # (B, slots_size, L)

            loss_intent = criterion_intents(intent, batch["intents"])
            loss_slot = criterion_slots(slots, batch["y_slots"])
            loss_array.append((loss_intent + loss_slot).item())

            # intent: argmax -> stringhe
            out_intents = [lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in batch["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # slot: argmax per token, poi sequenze (parola, etichetta) senza CLS
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = batch["slots_len"].tolist()[id_seq] - 1  # -1: esclude il CLS
                utt_ids = batch["utterances"][id_seq][:length].tolist()
                utterance = [lang.id2word[e] for e in utt_ids]
                gt_ids = batch["y_slots"][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[e] for e in gt_ids]
                to_decode = seq[:length].tolist()

                ref_slots.append([(utterance[j], gt_slots[j]) for j in range(length)])
                hyp_slots.append([(utterance[j], lang.id2slot[e]) for j, e in enumerate(to_decode)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # puo' capitare con uno slot predetto mai visto nel ground truth (lr alto, prime epoche)
        print(f"  Attenzione conll.evaluate: {ex}")
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


# ----------------------------------------------------------------------------
# 3. Training completo con early stopping (sulla slot F1 di dev)
# ----------------------------------------------------------------------------

def train_model(model, train_loader, dev_loader, lang, optimizer,
                criterion_slots, criterion_intents,
                n_epochs=200, patience=3, experiment_name="exp"):
    """Early stopping sulla slot F1 di dev (metrica piu' diretta della loss su un
    corpus piccolo come ATIS). Restituisce (best_model su CPU, best_dev_f1, best_dev_acc)."""
    best_f1 = 0.0
    best_acc = 0.0
    best_model = None
    cur_patience = patience

    pbar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")
    for epoch in pbar:
        train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        dev_results, dev_intent, _ = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )
        dev_f1 = dev_results["total"]["f"]
        dev_acc = dev_intent["accuracy"]
        pbar.set_postfix(dev_f1=f"{dev_f1:.3f}", intent_acc=f"{dev_acc:.3f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_acc = dev_acc
            best_model = copy.deepcopy(model).cpu()
            cur_patience = patience
        else:
            cur_patience -= 1
            if cur_patience <= 0:
                break

    if best_model is None:  # nessun miglioramento (raro): tieni l'ultimo
        best_model = copy.deepcopy(model).cpu()
    return best_model, best_f1, best_acc


# ----------------------------------------------------------------------------
# 4. Esperimento con piu' run (media +- std)
# ----------------------------------------------------------------------------

def run_experiments(train_loader, dev_loader, test_loader, lang,
                    vocab_len, slots_len, n_intents,
                    lr, d_model, n_heads, num_layers, ff_dim, dropout,
                    n_runs=5, n_epochs=200, patience=3,
                    experiment_name="exp", seed=42, device="cpu"):
    """n_runs training indipendenti (nuovo modello random ogni volta); riporta
    media +- std di slot F1 e intent accuracy sul TEST e la media della dev F1
    (usata per la selezione greedy). ATIS e' piccolo -> piu' run danno una stima
    affidabile."""
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    print(f"\n{'='*60}\nEsperimento: {experiment_name}")
    print(f"  lr={lr} d_model={d_model} n_heads={n_heads} "
          f"num_layers={num_layers} ff_dim={ff_dim} dropout={dropout}")
    print(f"  {n_runs} run x max {n_epochs} epoche\n{'='*60}")

    dev_f1s, dev_accs, test_f1s, test_accs = [], [], [], []
    n_params = None

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        torch.manual_seed(seed + run_idx)  # run diverse ma riproducibili

        model = GPT2(
            vocab_size=vocab_len, slots_size=slots_len, n_intents=n_intents,
            pos_emb_size=1024, d_model=d_model, n_heads=n_heads,
            num_layers=num_layers, ff_dim=ff_dim, dropout=dropout,
        ).to(device)
        model.apply(init_weights)
        if n_params is None:
            n_params = count_parameters(model)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        best_model, best_dev_f1, best_dev_acc = train_model(
            model, train_loader, dev_loader, lang, optimizer,
            criterion_slots, criterion_intents,
            n_epochs=n_epochs, patience=patience,
            experiment_name=f"{experiment_name}_run{run_idx+1}",
        )

        best_model = best_model.to(device)
        test_results, test_intent, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang
        )
        test_f1 = test_results["total"]["f"]
        test_acc = test_intent["accuracy"]
        print(f"  Run {run_idx+1}: dev F1={best_dev_f1:.4f} | "
              f"test F1={test_f1:.4f} test acc={test_acc:.4f}")

        dev_f1s.append(best_dev_f1)
        dev_accs.append(best_dev_acc)
        test_f1s.append(test_f1)
        test_accs.append(test_acc)

    dev_f1s, dev_accs = np.array(dev_f1s), np.array(dev_accs)
    test_f1s, test_accs = np.array(test_f1s), np.array(test_accs)
    print(f"\nRisultati {experiment_name} ({n_runs} run): "
          f"test Slot F1 {test_f1s.mean():.3f}+-{test_f1s.std():.3f} | "
          f"test Intent Acc {test_accs.mean():.3f}+-{test_accs.std():.3f}")

    return {
        "lr": lr, "d_model": d_model, "n_heads": n_heads,
        "num_layers": num_layers, "ff_dim": ff_dim, "dropout": dropout,
        "n_params": n_params, "n_runs": n_runs,
        "dev_f1_mean": round(float(dev_f1s.mean()), 4),
        "dev_acc_mean": round(float(dev_accs.mean()), 4),
        "slot_f1_mean": round(float(test_f1s.mean()), 4),
        "slot_f1_std": round(float(test_f1s.std()), 4),
        "intent_acc_mean": round(float(test_accs.mean()), 4),
        "intent_acc_std": round(float(test_accs.std()), 4),
        "slot_f1_runs": [round(float(x), 4) for x in test_f1s],
        "intent_acc_runs": [round(float(x), 4) for x in test_accs],
    }


# ----------------------------------------------------------------------------
# 5. Logging degli esperimenti (JSON) e report Markdown
# ----------------------------------------------------------------------------

def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def done_experiments(path):
    return {r["experiment"] for r in load_results(path)}


def append_result(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results = load_results(path)
    results.append(record)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def make_record(experiment, param, info, seed=42):
    """Costruisce il record JSON dell'esperimento (metriche dev per la selezione,
    metriche test per il report)."""
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "param": param,  # iperparametro variato in questo step
        "seed": seed,
    }
    rec.update(info)
    return rec


def regenerate_observations_md(results_path, md_path):
    """Rigenera il report Markdown. La selezione del migliore (per gruppo e globale)
    si fa sulla DEV slot F1; le metriche di test (media +- std) sono riportate."""
    results = load_results(results_path)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    lines = ["# Osservazioni esperimenti - NLU Part 2.A (intent + slot)", ""]
    lines.append(f"_Generato automaticamente da main.py. "
                 f"Ultimo aggiornamento: {datetime.now().isoformat(timespec='seconds')}._")
    lines.append("")
    lines.append("Metriche: **Slot F1** (conll) e **Intent Accuracy** sul test set "
                 "(media +- std su piu' run). Selezione degli iperparametri sulla **dev** slot F1.")
    lines.append("")

    if not results:
        lines.append("_Nessun esperimento registrato finora._")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    best = max(results, key=lambda r: r["dev_f1_mean"])
    lines.append("## Migliore configurazione finora")
    lines.append("")
    lines.append(f"- selezione su **dev F1: {best['dev_f1_mean']:.4f}** -> "
                 f"**test Slot F1: {best['slot_f1_mean']:.4f} +- {best['slot_f1_std']:.4f}**, "
                 f"**Intent Acc: {best['intent_acc_mean']:.4f} +- {best['intent_acc_std']:.4f}**")
    lines.append(f"- esperimento: `{best['experiment']}`")
    lines.append(f"- config: lr={best['lr']}, d_model={best['d_model']}, n_heads={best['n_heads']}, "
                 f"num_layers={best['num_layers']}, ff_dim={best['ff_dim']}, dropout={best['dropout']}")
    lines.append("")

    # raggruppa per iperparametro variato, mantenendo l'ordine di apparizione
    groups = {}
    for r in results:
        groups.setdefault(r.get("param", "-"), []).append(r)

    for g, runs in groups.items():
        lines.append(f"## {g}")
        lines.append("")
        lines.append("| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout "
                     "| dev F1 | test Slot F1 | test Intent Acc | |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        best_run = max(runs, key=lambda r: r["dev_f1_mean"])
        for r in runs:
            star = "*" if r is best_run else ""
            lines.append(
                f"| {r['experiment']} | {r['lr']} | {r['d_model']} | {r['n_heads']} "
                f"| {r['num_layers']} | {r['ff_dim']} | {r['dropout']} "
                f"| {r['dev_f1_mean']:.4f} | {r['slot_f1_mean']:.4f}+-{r['slot_f1_std']:.4f} "
                f"| {r['intent_acc_mean']:.4f}+-{r['intent_acc_std']:.4f} | {star} |"
            )
        lines.append("")
        lines.append(f"- Osservazione (scelta sul dev): migliore `{best_run['experiment']}` "
                     f"(dev F1 {best_run['dev_f1_mean']:.4f}, test Slot F1 {best_run['slot_f1_mean']:.4f}).")
        lines.append("- Note (da completare nel report): ")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
