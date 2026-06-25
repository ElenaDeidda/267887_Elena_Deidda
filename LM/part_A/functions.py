# functions.py
# Funzioni di supporto: loop di training, loop di valutazione (con Perplexity),
# inizializzazione dei pesi e una funzione "run_experiment" che incapsula un
# esperimento completo (costruzione modello -> training con early stopping ->
# valutazione sul test set). main.py usa queste funzioni.
#
# In piu' ci sono utility per il LOGGING degli esperimenti:
#   - append_result / load_results: salvano ogni run in results/results.json
#   - regenerate_observations_md:   rigenera results/observations.md (tabelle +
#                                   osservazioni automatiche) per aiutare a
#                                   scrivere il report.

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

# Soglia richiesta dalla consegna (PPL test deve stare sotto)
PPL_THRESHOLD = 250


def set_seed(seed=42):
    """Fissa i semi per rendere gli esperimenti riproducibili."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    """Numero totale di parametri (i pesi condivisi col weight tying contano una volta)."""
    return sum(p.numel() for p in model.parameters())


def train_loop(data, optimizer, criterion, model, clip=5.0):
    """Un'epoca di training. Restituisce la loss media pesata sui token."""
    model.train()
    loss_array = []
    number_of_tokens = []

    for input_ids, labels, n_tokens in data:
        optimizer.zero_grad()
        output = model(input_ids)                # (B, L, vocab)
        # CrossEntropyLoss vuole (B, vocab, L), quindi permutiamo
        loss = criterion(output.permute(0, 2, 1), labels)
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward()
        # gradient clipping: evita gradienti che esplodono nei transformer
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    """Valutazione senza calcolo del gradiente. Restituisce (PPL, loss media)."""
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, labels, n_tokens in data:
            output = model(input_ids)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)              # Perplexity = exp(cross-entropy)
    return ppl, loss_to_return


def init_weights(mat):
    """Inizializzazione uniforme dei layer lineari."""
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def run_experiment(name, config, loaders, tokenizer, device,
                   lr=5e-4, n_epochs=100, patience=3, init=True):
    """Esegue un esperimento completo e stampa la Perplexity finale sul test set.

    Args:
        name:    etichetta dell'esperimento (stampata a video)
        config:  dict con gli iperparametri del modello
                 (pos_emb_size, d_model, n_heads, num_layers, ff_dim,
                  dropout, weight_tying)
        loaders: tupla (train_loader, dev_loader, test_loader)
        lr:      learning rate per AdamW
        n_epochs, patience: parametri di training ed early stopping

    Returns:
        (best_model, info) dove info e' un dict con le metriche dell'esperimento:
        test_ppl, best_dev_ppl, best_epoch, epochs_run, n_params, config, lr.
    """
    train_loader, dev_loader, test_loader = loaders
    vocab_len = len(tokenizer)

    print(f"\n===== Esperimento: {name} =====")
    print(f"config: {config} | lr: {lr}")

    model = GPT2(vocab_len, **config).to(device)
    if init:
        # con il weight tying l'inizializzazione uniforme va bene comunque
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
        train_loop(train_loader, optimizer, criterion_train, model)
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, model)
        pbar.set_description(f"{name} | Dev PPL: {ppl_dev:.2f}")

        if ppl_dev < best_ppl:          # piu' bassa = migliore
            best_ppl = ppl_dev
            best_epoch = epoch
            best_model = copy.deepcopy(model).to('cpu')
            cur_patience = patience
        else:
            cur_patience -= 1
        if cur_patience <= 0:           # early stopping
            break

    best_model.to(device)
    test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
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
# Logging degli esperimenti (JSON) e generazione del report (Markdown)
# ----------------------------------------------------------------------------

def load_results(path):
    """Carica la lista dei run gia' salvati (lista vuota se il file non esiste)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def append_result(path, record):
    """Aggiunge un record di esperimento al file JSON (creandolo se serve)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results = load_results(path)
    results.append(record)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def make_record(group, label, info, swept_key=None, swept_value=None, seed=42):
    """Costruisce il dizionario da salvare nel JSON a partire dall'info di run_experiment."""
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
    """Converte valori non serializzabili (es. bool numpy) in tipi base."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


# Ordine canonico dei gruppi nel report (segue gli step della consegna)
_GROUP_ORDER = [
    "baseline-lr", "d_model", "n_heads", "num_layers", "ff_dim",
    "dropout", "weight_tying", "single",
]

# Descrizione di ogni gruppo, mostrata come intestazione nel report
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
    """Rigenera da zero il report Markdown a partire dal file JSON dei risultati.

    Per ogni gruppo produce una tabella ordinata, marca il run migliore con una
    stella e aggiunge un'osservazione automatica. In fondo riassume la migliore
    configurazione globale. Pensato per essere copiato/adattato nel report.
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

    # Migliore globale: la selezione si fa SEMPRE sul dev set (mai sul test).
    # Il test PPL viene solo riportato come numero finale della config scelta.
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

    # Raggruppa per gruppo, mantenendo l'ordine canonico
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
        best_run = min(runs, key=lambda r: r["best_dev_ppl"])  # scelta sul dev, non sul test
        for r in runs:
            star = "⭐" if r is best_run else ""
            val = r["swept_value"] if r["swept_value"] is not None else "-"
            ok = "sì" if r["passes_threshold"] else "**no**"
            lines.append(
                f"| {val} | {r['lr']} | {r['n_params']:,} | {r['epochs_run']} "
                f"| {r['best_dev_ppl']:.2f} | {r['test_ppl']:.2f} | {ok} | {star} |"
            )
        lines.append("")
        # Osservazione automatica
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
