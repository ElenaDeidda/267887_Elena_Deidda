# functions.py
# -----------------------------------------------------------------------------
# Training/evaluation engine for Part 1.A: the per-epoch training loop, the
# Perplexity evaluation loop, weight initialization, and run_experiment(),
# which wraps one full experiment (build model -> train with early stopping
# on dev PPL -> report test PPL). main.py drives these functions for the
# incremental hyperparameter search.
#
# Also includes experiment-LOGGING utilities:
#   - append_result / load_results: persist every run to results/results.json
#   - regenerate_observations_md:   regenerate results/observations.md (per-
#                                   sweep tables + auto-generated observations)
#                                   to help write up the report.

import copy
import json
import math
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import GPT2

# PPL threshold required by the assignment (test PPL must stay below this).
PPL_THRESHOLD = 250


def set_seed(seed=42):
    """Seed all RNGs used (Python random, torch CPU/CUDA) for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    """Total parameter count (weights shared via weight tying are counted once,
    since model.parameters() de-duplicates aliased tensors)."""
    return sum(p.numel() for p in model.parameters())


def train_loop(data, optimizer, criterion, model, device, clip=5.0):
    """Run one full training epoch over `data`.

    Args:
        data: train DataLoader yielding (input_ids, labels, n_tokens) batches
            from utils.collate_fn (still on CPU at this point).
        optimizer, criterion: AdamW optimizer and CrossEntropyLoss
            (ignore_index=pad_token_id) built by run_experiment.
        model: the GPT2 model.
        device: target device; the actual CPU->GPU transfer happens here
            (not in collate_fn - see utils.py for why).
        clip: max gradient norm for clipping.

    Returns:
        Token-weighted average training loss for the epoch (float). Weighting
        by n_tokens (rather than simply averaging per-batch losses) ensures
        batches with more non-pad tokens contribute proportionally more to
        the reported loss, so longer sequences aren't under-weighted relative
        to shorter/more-padded ones.
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    for input_ids, labels, n_tokens in data:
        # Device transfer happens here, in the main process, not in
        # collate_fn (which must stay CPU-only - see utils.py).
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(input_ids)                # (B, L, vocab)
        # CrossEntropyLoss expects (B, vocab, L) for sequence inputs, so permute.
        loss = criterion(output.permute(0, 2, 1), labels)
        # Accumulate loss*n_tokens (not just loss) so that summing across
        # batches and dividing by the total token count below yields a
        # correctly token-weighted average over the whole epoch.
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward()
        # Gradient clipping: prevents exploding gradients, a common issue
        # when training Transformers from scratch without warmup.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    # Normalize by total non-pad token count across the whole epoch, not by
    # number of batches, so the result is a true per-token average loss.
    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model, device):
    """Evaluate the model on `data` with gradients disabled.

    Same token-weighted loss averaging as train_loop (see that docstring for
    the rationale), used here to compute Perplexity = exp(average per-token
    cross-entropy loss).

    Returns:
        (ppl, avg_loss): Perplexity and the token-weighted average loss.
    """
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, labels, n_tokens in data:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(input_ids)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    # Token-weighted average loss over the whole dataset (see train_loop).
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)              # Perplexity = exp(cross-entropy)
    return ppl, loss_to_return


def init_weights(mat):
    """Initialize all nn.Linear layers with small uniform weights/biases.

    Applied via model.apply(init_weights) instead of relying on PyTorch's
    default initialization, for consistency/reproducibility across the many
    incremental experiments in the hyperparameter search.
    """
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def run_experiment(name, config, loaders, tokenizer, device,
                   lr=5e-4, n_epochs=100, patience=3, init=True):
    """Run one full experiment: build model, train with early stopping, report test PPL.

    Args:
        name:    experiment label (used for logging/progress bar).
        config:  dict of model hyperparameters forwarded to GPT2(**config)
                 (pos_emb_size, d_model, n_heads, num_layers, ff_dim,
                  dropout, weight_tying).
        loaders: tuple (train_loader, dev_loader, test_loader).
        lr:      learning rate for AdamW.
        n_epochs, patience: max epochs and early-stopping patience (see below).

    Model selection / early stopping policy (important for correctness of the
    reported results): the model is checkpointed (best_model) and patience is
    tracked based ONLY on dev set PPL, NEVER on test PPL. Test PPL is computed
    once at the very end, purely for reporting, and never influences which
    epoch/checkpoint is kept or when training stops. This avoids any leakage
    from the test set into model/hyperparameter selection.

    Returns:
        (best_model, info) where info is a dict of experiment metrics:
        test_ppl, best_dev_ppl, best_epoch, epochs_run, n_params, config, lr.
    """
    train_loader, dev_loader, test_loader = loaders
    vocab_len = len(tokenizer)

    print(f"\n===== Esperimento: {name} =====")
    print(f"config: {config} | lr: {lr}")

    model = GPT2(vocab_len, **config).to(device)
    if init:
        # Uniform initialization is applied the same way regardless of
        # weight_tying; when tying is enabled lm_head.weight and
        # token_embed.weight already point to the same tensor, so this still
        # initializes them correctly (as a single shared tensor).
        model.apply(init_weights)

    n_params = count_params(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_ppl = math.inf
    best_model = None
    best_epoch = -1
    epochs_run = 0
    cur_patience = patience

    pbar = tqdm(range(n_epochs), desc=name)
    for epoch in pbar:
        epochs_run = epoch + 1
        train_loop(train_loader, optimizer, criterion_train, model, device)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model, device)
        pbar.set_description(f"{name} | Dev PPL: {ppl_dev:.2f}")

        if ppl_dev < best_ppl:          # lower PPL = better
            best_ppl = ppl_dev
            best_epoch = epoch
            # Snapshot the best model on CPU so GPU memory isn't held by
            # multiple full copies of the model across epochs.
            best_model = copy.deepcopy(model).to('cpu')
            cur_patience = patience
        else:
            cur_patience -= 1
        if cur_patience <= 0:           # early stopping: dev PPL hasn't
            break                       # improved for `patience` epochs in a row

    best_model.to(device)
    # Test PPL is computed only here, after model selection is already final
    # (based on dev PPL above) - purely for reporting, never for selection.
    test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model, device)
    print(f"[{name}] Best Dev PPL: {best_ppl:.2f} | Test PPL: {test_ppl:.2f} "
          f"| params: {n_params:,} | epoche: {epochs_run} (best @ {best_epoch})")

    info = {
        "best_dev_ppl": round(float(best_ppl), 4),
        "test_ppl": round(float(test_ppl), 4),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "n_params": n_params,
        "config": dict(config),
        "lr": lr,
    }
    return best_model, info


# ----------------------------------------------------------------------------
# Experiment logging (JSON) and Markdown report generation
# ----------------------------------------------------------------------------

def load_results(path):
    """Load the list of previously saved experiment runs (empty list if the
    results file doesn't exist yet)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def append_result(path, record):
    """Append one experiment record to the JSON results file, creating the
    file (and its parent directory) if it doesn't exist yet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results = load_results(path)
    results.append(record)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def make_record(group, label, info, swept_key=None, swept_value=None, seed=42):
    """Build the JSON-serializable record for one run from run_experiment's info dict."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "group": group,
        "label": label,
        "swept_key": swept_key,
        "swept_value": _jsonable(swept_value),
        "lr": info["lr"],
        "config": info["config"],
        "n_params": info["n_params"],
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "best_dev_ppl": info["best_dev_ppl"],
        "test_ppl": info["test_ppl"],
        "passes_threshold": bool(info["test_ppl"] < PPL_THRESHOLD),
        "seed": seed,
    }


def _jsonable(v):
    """Coerce a value into something json.dump can serialize (e.g. numpy bools)."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


# Canonical ordering of sweep groups in the generated report, mirroring the
# steps of the incremental hyperparameter search required by the assignment.
_GROUP_ORDER = [
    "baseline-lr", "d_model", "n_heads", "num_layers", "ff_dim",
    "dropout", "weight_tying", "single",
]

# Human-readable description of each group, shown as a section header in the
# generated Markdown report.
_GROUP_DESC = {
    "baseline-lr": "Step 0 - Baseline: ricerca del learning rate (modello fisso).",
    "d_model":     "Step 1 - Iperparametri: dimensione del modello (d_model).",
    "n_heads":     "Step 1 - Iperparametri: numero di teste di attenzione (n_heads).",
    "num_layers":  "Step 1 - Iperparametri: numero di blocchi transformer (num_layers).",
    "ff_dim":      "Step 1 - Iperparametri: dimensione del feed-forward (ff_dim).",
    "dropout":     "Step 2 - Dropout nei 4 punti della rete.",
    "weight_tying": "Step 3 - Weight tying tra token_embed e lm_head.",
    "single":      "Esperimenti singoli / configurazione finale.",
}


def regenerate_observations_md(results_path, md_path):
    """Regenerate the Markdown observations report from scratch from the JSON results log.

    For each sweep group, produces a sorted table, marks the best run (chosen
    by dev PPL, see below) with a star, and appends an auto-generated
    observation sentence. At the end, summarizes the single best configuration
    found so far across all groups. Intended to be copy/pasted (or adapted)
    directly into the written report, rather than maintained by hand.
    """
    results = load_results(results_path)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    lines = []
    lines.append("# Osservazioni esperimenti - LM Part 1.A")
    lines.append("")
    lines.append(f"_Generato automaticamente da main.py. "
                 f"Ultimo aggiornamento: {datetime.now().isoformat(timespec='seconds')}._")
    lines.append("")
    lines.append(f"Vincolo della consegna: **PPL test < {PPL_THRESHOLD}**. "
                 "La modifica va tenuta solo se migliora (o non peggiora) la PPL; "
                 "gli esperimenti falliti vanno comunque commentati nel report.")
    lines.append("")

    if not results:
        lines.append("_Nessun esperimento registrato finora._")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    # Global best: selection is ALWAYS done on the dev set (never on test).
    # Test PPL is only reported as the final number for the chosen config.
    best = min(results, key=lambda r: r["best_dev_ppl"])
    lines.append("## Migliore configurazione finora")
    lines.append("")
    lines.append(f"- selezione su **best dev PPL: {best['best_dev_ppl']:.2f}** "
                 f"-> **Test PPL: {best['test_ppl']:.2f}** "
                 f"({'OK <250' if best['passes_threshold'] else 'NON soddisfa <250'})")
    lines.append(f"- gruppo: `{best['group']}` | label: `{best['label']}`")
    lines.append(f"- lr: `{best['lr']}` | parametri: {best['n_params']:,}")
    lines.append(f"- config: `{best['config']}`")
    lines.append("")

    # Group records by sweep group, preserving the canonical order defined above.
    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)
    ordered = [g for g in _GROUP_ORDER if g in groups]
    ordered += [g for g in groups if g not in _GROUP_ORDER]

    for g in ordered:
        runs = groups[g]
        lines.append(f"## {g}")
        desc = _GROUP_DESC.get(g)
        if desc:
            lines.append(f"_{desc}_")
        lines.append("")
        lines.append("| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |")
        lines.append("|---|---|---|---|---|---|---|---|")
        best_run = min(runs, key=lambda r: r["best_dev_ppl"])  # selected by dev PPL, not test
        for r in runs:
            star = "⭐" if r is best_run else ""
            val = r["swept_value"] if r["swept_value"] is not None else "-"
            ok = "sì" if r["passes_threshold"] else "**no**"
            lines.append(
                f"| {val} | {r['lr']} | {r['n_params']:,} | {r['epochs_run']} "
                f"| {r['best_dev_ppl']:.2f} | {r['test_ppl']:.2f} | {ok} | {star} |"
            )
        lines.append("")
        # Auto-generated observation sentence summarizing this group's spread.
        if len(runs) > 1 and best_run["swept_value"] is not None:
            worst_run = max(runs, key=lambda r: r["best_dev_ppl"])
            delta = worst_run["best_dev_ppl"] - best_run["best_dev_ppl"]
            lines.append(
                f"- Osservazione: il valore migliore (scelto sul dev) e' "
                f"**{best_run['swept_value']}** (dev PPL {best_run['best_dev_ppl']:.2f}, "
                f"test PPL {best_run['test_ppl']:.2f}); il peggiore "
                f"{worst_run['swept_value']} (dev PPL {worst_run['best_dev_ppl']:.2f}), "
                f"un divario di {delta:.2f} punti di dev PPL."
            )
        else:
            lines.append(
                f"- Osservazione: test PPL {best_run['test_ppl']:.2f} "
                f"({'soddisfa' if best_run['passes_threshold'] else 'NON soddisfa'} il vincolo <250)."
            )
        lines.append("- Note (da completare nel report): ")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
