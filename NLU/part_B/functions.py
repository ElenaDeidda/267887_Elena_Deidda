# functions.py
# Part 2.B - Training e valutazione per il fine-tuning di BERT/GPT-2, piu' il
# logging degli esperimenti in JSON (come le altre parti).
#
# Differenze rispetto alla Part 2.A:
#   - ignore_index = -100 (token speciali, sub-token dopo il primo, padding).
#   - La decodifica slot usa la maschera (y_slots != -100) per selezionare il
#     primo sub-token di ogni parola (invece di slots_len - 1); le parole originali
#     arrivano da batch['words'].
#   - Il batch mescola tensori e liste Python ('words'): .to(device) solo sui tensori.
#   - model_type ('bert'/'gpt2') sceglie la firma del forward.

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

from model import BERTforNLU, GPT2forNLU, count_parameters

IGNORE_INDEX = -100


def _to_device(batch, device):
    """Sposta su device solo i tensori (lascia 'words', che e' una lista Python)."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _forward(model, batch, model_type):
    """Chiama il forward giusto in base al tipo di modello."""
    if model_type == "bert":
        return model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
    return model(batch["input_ids"], batch["attention_mask"], batch["seq_lens"])


# ----------------------------------------------------------------------------
# 1. Loop di training (una epoca)
# ----------------------------------------------------------------------------

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, model_type="bert"):
    """Una epoca di training (BERT o GPT-2). Restituisce la lista delle loss per batch."""
    model.train()
    loss_array = []
    device = next(model.parameters()).device

    for batch in tqdm(data, desc="  train", leave=False):
        optimizer.zero_grad()
        batch = _to_device(batch, device)

        slots, intent = _forward(model, batch, model_type)
        slots = slots.permute(0, 2, 1)  # (B, L, slots_size) -> (B, slots_size, L)

        loss = criterion_intents(intent, batch["intents"]) + \
               criterion_slots(slots, batch["y_slots"])
        loss_array.append(loss.item())

        loss.backward()
        optimizer.step()
    return loss_array


# ----------------------------------------------------------------------------
# 2. Loop di valutazione (dev o test)
# ----------------------------------------------------------------------------

def eval_loop(data, criterion_slots, criterion_intents, model, lang, model_type="bert"):
    """Valuta senza aggiornare i pesi. Gli slot reali sono le posizioni con
    (y_slots != -100) (primo sub-token di ogni parola); le parole originali sono
    in batch['words'].

    Returns: results (conll), report_intent (classification_report), loss_array.
    """
    model.eval()
    loss_array = []
    device = next(model.parameters()).device

    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []

    with torch.no_grad():
        for batch in tqdm(data, desc="  eval", leave=False):
            batch = _to_device(batch, device)

            slots, intent = _forward(model, batch, model_type)
            slots = slots.permute(0, 2, 1)  # (B, slots_size, L)

            loss = criterion_intents(intent, batch["intents"]) + \
                   criterion_slots(slots, batch["y_slots"])
            loss_array.append(loss.item())

            out_intents = [lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in batch["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # slot: tieni solo i primi sub-token (mask != -100), allinea alle parole
            output_slots = torch.argmax(slots, dim=1)  # (B, L)
            for id_seq in range(output_slots.size(0)):
                words = batch["words"][id_seq]
                y = batch["y_slots"][id_seq]      # -100 nelle posizioni ignorate
                preds = output_slots[id_seq]
                mask = (y != IGNORE_INDEX)

                real_gt_ids = y[mask].tolist()
                real_pred_ids = preds[mask].tolist()

                n_real = len(real_gt_ids)          # < len(words) se troncato
                words_aligned = words[:n_real]

                ref_slots.append([(words_aligned[j], lang.id2slot[real_gt_ids[j]])
                                   for j in range(n_real)])
                hyp_slots.append([(words_aligned[j], lang.id2slot[real_pred_ids[j]])
                                   for j in range(n_real)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
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
                model_type="bert", n_epochs=30, patience=3, experiment_name="exp"):
    """Early stopping sulla slot F1 di dev. Il fine-tuning converge in poche epoche.
    Restituisce (best_model su CPU, best_dev_f1, best_dev_acc)."""
    best_f1 = 0.0
    best_acc = 0.0
    best_model = None
    cur_patience = patience

    pbar = tqdm(range(1, n_epochs + 1), desc=f"[{experiment_name}]", unit="ep")
    for epoch in pbar:
        train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, model_type)
        dev_results, dev_intent, _ = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang, model_type
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

    if best_model is None:
        best_model = copy.deepcopy(model).cpu()
    return best_model, best_f1, best_acc


# ----------------------------------------------------------------------------
# 4. Esperimento con piu' run (media +- std)
# ----------------------------------------------------------------------------

def run_experiments(make_datasets, lang, slots_size, n_intents,
                    lr, model_name, model_type="bert", dropout=0.1,
                    n_runs=3, n_epochs=30, patience=3,
                    experiment_name="exp", seed=42, device="cpu"):
    """n_runs fine-tuning indipendenti e media +- std. `make_datasets()` ricrea i
    tre loader (utile per il seed). Riporta anche la media della dev F1 (selezione)."""
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_intents = nn.CrossEntropyLoss()

    print(f"\n{'='*65}\nEsperimento: {experiment_name}")
    print(f"  model={model_name} ({model_type}) | lr={lr} dropout={dropout}")
    print(f"  {n_runs} run x max {n_epochs} epoche\n{'='*65}")

    dev_f1s, dev_accs, test_f1s, test_accs = [], [], [], []
    n_params = None

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        torch.manual_seed(seed + run_idx)
        train_loader, dev_loader, test_loader = make_datasets()

        if model_type == "bert":
            model = BERTforNLU(slots_size, n_intents, model_name, dropout).to(device)
        else:
            model = GPT2forNLU(slots_size, n_intents, model_name, dropout).to(device)
        if n_params is None:
            n_params = count_parameters(model)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        best_model, best_dev_f1, best_dev_acc = train_model(
            model, train_loader, dev_loader, lang, optimizer,
            criterion_slots, criterion_intents, model_type=model_type,
            n_epochs=n_epochs, patience=patience,
            experiment_name=f"{experiment_name}_run{run_idx+1}",
        )

        best_model = best_model.to(device)
        test_results, test_intent, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, best_model, lang, model_type
        )
        test_f1 = test_results["total"]["f"]
        test_acc = test_intent["accuracy"]
        print(f"  Run {run_idx+1}: dev F1={best_dev_f1:.4f} | test F1={test_f1:.4f} acc={test_acc:.4f}")

        dev_f1s.append(best_dev_f1)
        dev_accs.append(best_dev_acc)
        test_f1s.append(test_f1)
        test_accs.append(test_acc)

        del model, best_model, optimizer, train_loader, dev_loader, test_loader
        if device == "cuda":
            torch.cuda.empty_cache()

    dev_f1s, dev_accs = np.array(dev_f1s), np.array(dev_accs)
    test_f1s, test_accs = np.array(test_f1s), np.array(test_accs)
    print(f"\nRisultati {experiment_name} ({n_runs} run): "
          f"test Slot F1 {test_f1s.mean():.3f}+-{test_f1s.std():.3f} | "
          f"test Intent Acc {test_accs.mean():.3f}+-{test_accs.std():.3f}")

    return {
        "model_name": model_name, "model_type": model_type,
        "lr": lr, "dropout": dropout, "n_params": n_params, "n_runs": n_runs,
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
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment, "param": param, "seed": seed,
    }
    rec.update(info)
    return rec


def regenerate_observations_md(results_path, md_path):
    """Rigenera il report Markdown, raggruppando per modello (BERT vs GPT-2).
    Selezione del migliore sulla DEV F1; metriche di test (media +- std) riportate."""
    results = load_results(results_path)
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)

    lines = ["# Osservazioni esperimenti - NLU Part 2.B (fine-tuning BERT/GPT-2)", ""]
    lines.append(f"_Generato automaticamente da main.py. "
                 f"Ultimo aggiornamento: {datetime.now().isoformat(timespec='seconds')}._")
    lines.append("")
    lines.append("Metriche: **Slot F1** (conll) e **Intent Accuracy** sul test set "
                 "(media +- std). Selezione degli iperparametri sulla **dev** slot F1.")
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
    lines.append(f"- esperimento: `{best['experiment']}` "
                 f"(modello `{best['model_name']}`, lr={best['lr']}, dropout={best['dropout']})")
    lines.append("")

    # raggruppa per modello
    groups = {}
    for r in results:
        groups.setdefault(r.get("model_type", "-"), []).append(r)

    for g, runs in groups.items():
        lines.append(f"## {g}")
        lines.append("")
        lines.append("| esperimento | model | lr | dropout | dev F1 | test Slot F1 | test Intent Acc | |")
        lines.append("|---|---|---|---|---|---|---|---|")
        best_run = max(runs, key=lambda r: r["dev_f1_mean"])
        for r in runs:
            star = "*" if r is best_run else ""
            lines.append(
                f"| {r['experiment']} | {r['model_name']} | {r['lr']} | {r['dropout']} "
                f"| {r['dev_f1_mean']:.4f} | {r['slot_f1_mean']:.4f}+-{r['slot_f1_std']:.4f} "
                f"| {r['intent_acc_mean']:.4f}+-{r['intent_acc_std']:.4f} | {star} |"
            )
        lines.append("")
        lines.append(f"- Osservazione (scelta sul dev): migliore `{best_run['experiment']}` "
                     f"(dev F1 {best_run['dev_f1_mean']:.4f}, test Slot F1 {best_run['slot_f1_mean']:.4f}, "
                     f"Intent Acc {best_run['intent_acc_mean']:.4f}).")
        lines.append("- Note (da completare nel report): ")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
