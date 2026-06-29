# functions.py
# Part 1.B - Backbone freezing + LoRA adapter enabling, training/eval loops, early
# stopping, and JSON experiment logging (mirrors the structure of Part 1.A).
#
# Key difference from Part 1.A: there, the loss was computed manually outside the
# model after manually shifting logits/labels by one position. Here, the loss is
# computed INSIDE GPT2LMHeadModel: passing `labels` to the model's forward call makes
# it return `output.loss`, with the input/label shift performed internally by
# HuggingFace. Padding positions are masked out of that internal loss by setting them
# to -100 before calling the model (PyTorch's cross-entropy convention: -100 is
# ignored). Manually shifting labels in this file, as Part 1.A does, would therefore be
# redundant -- and in fact wrong, since the model would then shift an already-shifted
# tensor. Only the LoRA adapter matrices are trainable; the rest of GPT-2 is frozen via
# freeze_pretrained_and_enable_lora below.

import os
import json
import math
import copy
from datetime import datetime

import torch
from tqdm import tqdm

from model import CustomGPT2Attention

# Assignment requirement: test PPL must be below 250 (and lower than Part 1.A's
# from-scratch baseline).
PPL_THRESHOLD = 250


# ----------------------------------------------------------------------------
# 1. Freeze the pre-trained backbone / enable only the LoRA adapters
# ----------------------------------------------------------------------------

def freeze_pretrained_and_enable_lora(model):
    """Freeze every parameter in `model`, then re-enable gradients only for the LoRA
    matrices inside each CustomGPT2Attention module.

    This two-pass approach (freeze everything, then selectively unfreeze) is simpler
    and less error-prone than trying to freeze only "the backbone" by name matching,
    and it guarantees by construction that the ~124M pre-trained GPT-2 parameters never
    receive gradients or get updated by the optimizer -- only the small LoRA adapter
    matrices (lora_A_*/lora_B_* on Q, K, V) do.
    """
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
# 2. Training loop (one epoch)
# ----------------------------------------------------------------------------

def train_loop(data, optimizer, model, tokenizer, clip=5.0):
    """Run one training epoch over `data`.

    Returns the token-weighted average cross-entropy loss over the whole epoch (i.e.
    sum of per-batch total loss divided by total token count, not a plain average of
    per-batch means -- this is correct under variable batch lengths/sizes).
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="  train", leave=False)
    for input_ids, _labels, n_tokens in pbar:
        optimizer.zero_grad()

        # Build labels = input_ids with padding positions set to -100. We do NOT shift
        # these labels by one position ourselves: GPT2LMHeadModel.forward() performs
        # the input/label shift internally when `labels` is passed, and ignores any
        # position whose label equals -100 when computing the cross-entropy loss. The
        # `_labels` tensor returned by collate_fn (which IS pre-shifted) is therefore
        # deliberately discarded here in favor of this internally-shifted approach.
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100

        output = model(input_ids, labels=labels)  # logits + internally-computed loss

        # output.loss is the mean loss per (non-ignored) token; multiplying back by
        # n_tokens recovers the batch's total loss so it can be aggregated correctly
        # across batches of different sizes below.
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)

        output.loss.backward()
        # Gradient clipping over all parameters: gradients only flow into the LoRA
        # matrices (the frozen backbone parameters have requires_grad=False and
        # therefore no .grad to clip), but clip_grad_norm_ is called on the full
        # parameter set for simplicity since it's a no-op on frozen tensors anyway.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)


# ----------------------------------------------------------------------------
# 3. Evaluation loop
# ----------------------------------------------------------------------------

def eval_loop(data, model, tokenizer):
    """Evaluate `model` on `data` (dev or test) without updating any weights.

    Returns (perplexity, average_loss_per_token).
    """
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, _labels, n_tokens in tqdm(data, desc="  eval", leave=False):
            # Same -100-masking / internal-shift pattern as train_loop (see comments
            # there); kept consistent so train and eval loss are computed identically.
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100

            output = model(input_ids, labels=labels)

            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_avg = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(min(loss_avg, 100))  # cap at e^100 to avoid overflow during early epochs
    return ppl, loss_avg


# ----------------------------------------------------------------------------
# 4. Full training run with early stopping (on dev PPL)
# ----------------------------------------------------------------------------

def train_model(model, train_loader, dev_loader, tokenizer, optimizer,
                n_epochs=20, patience=3, experiment_name="exp"):
    """Train `model` for up to `n_epochs`, with early stopping based on dev PPL.

    Because the backbone is already pre-trained and only a small LoRA adapter is being
    fit, convergence is typically much faster than training from scratch (Part 1.A) --
    often within a handful of epochs.

    Returns:
        best_model (moved to CPU), best_dev_ppl, best_epoch, epochs_run
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
    """Count trainable parameters (the LoRA matrices only, once frozen)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# 5. Experiment logging (JSON) and Markdown report generation - mirrors Part 1.A
# ----------------------------------------------------------------------------

def load_results(path):
    """Load the list of previously saved experiment records (empty list if the results
    file doesn't exist yet)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def done_experiments(path):
    """Return the set of experiment names already present in the results JSON.

    Used by main.py to make the greedy search idempotent: any experiment whose name is
    already in this set is skipped rather than re-run, so interrupting and restarting
    the script (e.g. after a VM reboot) resumes safely instead of duplicating runs.
    """
    return {r["experiment"] for r in load_results(path)}


def append_result(path, record):
    """Append one experiment record to the JSON results file, creating the file (and
    its parent directory) if it doesn't exist yet."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results = load_results(path)
    results.append(record)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def make_record(experiment, step, info, seed=42):
    """Build the JSON-serializable dict describing one experiment run, from the raw
    metrics gathered in main.run_one."""
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
    """Regenerate the human-readable Markdown report from the JSON results file.

    The "best" configuration (both per-step and overall) is ALWAYS selected by
    minimum dev PPL, never by test PPL -- test PPL is only reported alongside the
    winning configuration as the final number, to avoid implicitly tuning
    hyperparameters against the test set.
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
