# functions.py
# Part 2.A - Training and evaluation loops for the joint intent classification
# + slot filling model, plus JSON experiment logging (mirrors Part 1.A).
#
# Joint loss = CE(intent) + CE(slot), with the slot cross-entropy using
# ignore_index=PAD_TOKEN so that BOTH padding positions AND the CLS position
# (which is assigned the same id as PAD in slot2id, see utils.py) contribute
# zero gradient/loss for slot filling -- only genuine word positions are
# scored. Slots are evaluated with chunk-level (span-level) F1 via conll.py,
# rather than naive per-token accuracy, because a multi-word slot (e.g. a
# city name spanning several BIO-tagged tokens) is only counted as correct if
# the WHOLE span matches; this is the standard CoNLL-2000 chunking evaluation
# convention. Intents are evaluated with plain accuracy. Each configuration
# is repeated over multiple runs (mean +- std) to average out the
# variance from random initialization/seed effects, and early stopping
# tracks dev slot F1 with a deliberately low patience (3 epochs without
# improvement), since this is a small model on a small dataset that
# converges quickly -- a long patience would just waste epochs.

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
# 1. Training loop (one epoch)
# ----------------------------------------------------------------------------

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """Run one training epoch over `data` and return the list of per-batch
    joint loss values (used only for logging/inspection, not for early
    stopping, which is based on dev slot F1 instead -- see train_model)."""
    model.train()
    loss_array = []
    for batch in tqdm(data, desc="  train", leave=False):
        optimizer.zero_grad()
        slots, intent = model(batch["utterances"], batch["slots_len"])
        # model() returns slot logits as (B, L, n_slots), but
        # nn.CrossEntropyLoss for a per-token multi-class target expects the
        # class dimension in position 1, i.e. (B, C, L). permute(0, 2, 1)
        # reorders the axes accordingly without changing any values.
        slots = slots.permute(0, 2, 1)  # (B, L, n_slots) -> (B, n_slots, L)

        loss_intent = criterion_intents(intent, batch["intents"])
        loss_slot = criterion_slots(slots, batch["y_slots"])
        loss = loss_intent + loss_slot  # joint loss, intent and slot terms weighted equally
        loss_array.append(loss.item())

        loss.backward()
        optimizer.step()
    return loss_array


# ----------------------------------------------------------------------------
# 2. Evaluation loop (dev or test)
# ----------------------------------------------------------------------------

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """Evaluate the model without updating weights: decode predicted intents
    and slot sequences back to strings and compute metrics.

    Slot decoding uses length = slots_len[i] - 1 to EXCLUDE the trailing CLS
    position, because conll.evaluate() expects plain word/label sequences
    (it has no notion of a CLS token, and including it would corrupt the
    chunk boundaries it computes).

    Returns:
        results: dict returned by conll.evaluate(); results['total']['f'] is
            the chunk-level (span-level) slot F1.
        report_intent: dict from sklearn's classification_report;
            report_intent['accuracy'] is the intent accuracy.
        loss_array: list of per-batch joint loss values (logging only).
    """
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []

    with torch.no_grad():
        for batch in tqdm(data, desc="  eval", leave=False):
            slots, intent = model(batch["utterances"], batch["slots_len"])
            # Same axis reorder as in train_loop: CrossEntropyLoss needs the
            # class dimension at index 1, i.e. (B, slots_size, L).
            slots = slots.permute(0, 2, 1)  # (B, slots_size, L)

            loss_intent = criterion_intents(intent, batch["intents"])
            loss_slot = criterion_slots(slots, batch["y_slots"])
            loss_array.append((loss_intent + loss_slot).item())

            # Intent: take the argmax class per example and map back to strings.
            out_intents = [lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in batch["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slots: take the argmax label per token, then build (word, label)
            # sequences for conll, truncated to exclude the CLS token/position.
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = batch["slots_len"].tolist()[id_seq] - 1  # -1: drop the trailing CLS position
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
        # Can happen if the model predicts a slot label never seen in this
        # batch's ground truth (e.g. early epochs with a high learning rate);
        # conll's internal bookkeeping can raise on certain label patterns,
        # so we degrade gracefully to F1=0 rather than crashing the whole run.
        print(f"  Attenzione conll.evaluate: {ex}")
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


# ----------------------------------------------------------------------------
# 3. Full training with early stopping (on dev slot F1)
# ----------------------------------------------------------------------------

def train_model(model, train_loader, dev_loader, lang, optimizer,
                criterion_slots, criterion_intents,
                n_epochs=200, patience=3, experiment_name="exp"):
    """Train for up to n_epochs with early stopping on dev slot F1.

    Dev slot F1 is used as the early-stopping criterion (rather than dev
    loss) because it is the more direct/interpretable measure of what we
    actually care about on a small corpus like ATIS. `patience` is
    deliberately small (default 3): this is a small model on a small
    dataset that converges fast, so a long patience would mostly just train
    extra epochs without improving the selected checkpoint.

    Returns:
        best_model: deep copy of the model (moved to CPU) from the epoch
            with the highest dev slot F1 seen so far.
        best_f1: that epoch's dev slot F1.
        best_acc: that epoch's dev intent accuracy.
    """
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

    if best_model is None:  # no improvement at all over n_epochs (rare): keep the last epoch's model
        best_model = copy.deepcopy(model).cpu()
    return best_model, best_f1, best_acc


# ----------------------------------------------------------------------------
# 4. Multi-run experiment (mean +- std)
# ----------------------------------------------------------------------------

def run_experiments(train_loader, dev_loader, test_loader, lang,
                    vocab_len, slots_len, n_intents,
                    lr, d_model, n_heads, num_layers, ff_dim, dropout,
                    n_runs=5, n_epochs=200, patience=3,
                    experiment_name="exp", seed=42, device="cpu"):
    """Run n_runs independent trainings of one hyperparameter configuration
    (a freshly re-initialized model each time, seeded as seed+run_idx) and
    aggregate the results.

    Running multiple seeds matters specifically because ATIS is small: a
    single run's test/dev score can vary noticeably just due to random
    initialization and minibatch ordering, so averaging over n_runs (default
    5) gives a much more reliable estimate of whether a candidate
    hyperparameter value is actually better, rather than picking a value
    that merely got a lucky seed (this is also why main.py's greedy search
    always compares the MEAN dev F1 across n_runs, not a single run's score,
    before deciding whether to keep a candidate).

    Returns a dict with the config, parameter count, and mean+-std of dev F1,
    dev intent accuracy, test slot F1, and test intent accuracy (the dev F1
    mean is what main.py's run_search uses to pick the winner of each step;
    the test metrics are what gets reported in the final results/paper).
    """
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
        torch.manual_seed(seed + run_idx)  # different but reproducible seed per run

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
# 5. Experiment logging (JSON) and Markdown report generation
# ----------------------------------------------------------------------------
#
# This JSON file (results/results.json) is what makes main.py's search
# idempotent: each experiment is keyed by its unique name, and done_experiments
# is used by main.py to skip any experiment that has already been recorded
# (so an interrupted/resumed search does not redundantly retrain configs).

def load_results(path):
    """Load the list of experiment records from `path`, or an empty list if
    the results file does not exist yet."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def done_experiments(path):
    """Return the set of experiment names already present in the results
    file, used by main.py to decide whether a candidate needs (re)training."""
    return {r["experiment"] for r in load_results(path)}


def append_result(path, record):
    """Append one experiment record to the JSON results file (read-modify-write;
    creates the parent directory and file if they do not exist yet)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results = load_results(path)
    results.append(record)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def make_record(experiment, param, info, seed=42):
    """Build the JSON record for one experiment: dev metrics are what main.py
    uses to select the winning candidate at each search step; test metrics
    (mean +- std across runs) are what gets reported in the final results."""
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "param": param,  # which hyperparameter was varied in this search step
        "seed": seed,
    }
    rec.update(info)
    return rec


def regenerate_observations_md(results_path, md_path):
    """Regenerate the human-readable Markdown report from the JSON results.
    The "best" selection (both per-group and overall) is always made on DEV
    slot F1; test metrics (mean +- std) are reported for reference but never
    used to pick a winner, to avoid implicitly tuning on the test set."""
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
