# Generates the new NLU error-analysis figures and refreshes nlu_params_vs_f1.png.
#   nlu_perslot_f1.png       - per-slot-type F1, BERT vs GPT-2-medium (test ATIS)
#   nlu_intent_confusion.png - intent confusion matrix (recall) for best BERT
#   nlu_params_vs_f1.png     - refreshed: fixed overlapping point labels
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RT = os.path.dirname(os.path.abspath(__file__))          # report_template/
ROOT = os.path.dirname(RT)                                # project root
ANALYSIS = json.load(open(os.path.join(RT, "nlu_analysis.json")))
RES_A = json.load(open(os.path.join(ROOT, "NLU", "part_A", "results", "results.json")))
RES_B = json.load(open(os.path.join(ROOT, "NLU", "part_B", "results", "results.json")))

TEAL = "#2a9d8f"    # BERT
ORANGE = "#e8833a"  # GPT-2
BLUE = "#4c72b0"    # from-scratch

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# 1. Per-slot F1: BERT vs GPT-2-medium
# ---------------------------------------------------------------------------
def fig_perslot():
    ps_b = ANALYSIS["bert"]["perslot"]
    ps_g = ANALYSIS["gpt2"]["perslot"]
    # reliable slots: support >= 20, sorted by support (descending)
    slots = [(k, v["s"], v["f"], ps_g.get(k, {}).get("f", 0.0))
             for k, v in ps_b.items() if k != "total" and v["s"] >= 20]
    slots.sort(key=lambda x: x[1])  # ascending -> most frequent ends on top
    slots = slots[-14:]
    names = [f"{k}  (n={s})" for k, s, fb, fg in slots]
    fb = [x[2] for x in slots]
    fg = [x[3] for x in slots]

    y = np.arange(len(slots))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.barh(y + h / 2, fb, height=h, color=TEAL, label="BERT-base", zorder=3)
    ax.barh(y - h / 2, fg, height=h, color=ORANGE, label="GPT-2-medium", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test Slot F1 (chunk-level, CoNLL)")
    ax.set_title("Per-slot F1: where bidirectional context helps")
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2,
              frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(os.path.join(RT, "nlu_perslot_f1.png"))
    plt.close(fig)
    print("wrote nlu_perslot_f1.png")


# ---------------------------------------------------------------------------
# 2. Intent confusion matrix (recall) for best BERT
# ---------------------------------------------------------------------------
def fig_confusion():
    ref = ANALYSIS["bert"]["ref_int"]
    hyp = ANALYSIS["bert"]["hyp_int"]
    from collections import Counter
    support = Counter(ref)
    pair = Counter((r, h) for r, h in zip(ref, hyp) if r != h)

    # label set: top intents by support, plus any label in a >=2 confusion pair
    labels = [k for k, _ in support.most_common(9)]
    for (r, h), c in pair.items():
        if c >= 2:
            for lab in (r, h):
                if lab not in labels:
                    labels.append(lab)
    # order by support (desc), unknown-support preds last
    labels = sorted(set(labels), key=lambda l: -support.get(l, 0))

    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    counts = np.zeros((n, n))
    for r, h in zip(ref, hyp):
        if r in idx and h in idx:
            counts[idx[r], idx[h]] += 1
    row_tot = np.array([support[l] for l in labels], dtype=float)
    recall = counts / row_tot[:, None]

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(recall, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels([f"{l}  (n={support[l]})" for l in labels])
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title("Intent confusion — best BERT (recall, top intents)")
    for i in range(n):
        for j in range(n):
            c = int(counts[i, j])
            if c == 0:
                continue
            ax.text(j, i, str(c), ha="center", va="center",
                    color="white" if recall[i, j] > 0.55 else "black",
                    fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (recall)")
    fig.savefig(os.path.join(RT, "nlu_intent_confusion.png"))
    plt.close(fig)
    print(f"wrote nlu_intent_confusion.png  (errors={sum(pair.values())}/{len(ref)})")


# ---------------------------------------------------------------------------
# 3. Refresh params-vs-F1 with non-overlapping labels
# ---------------------------------------------------------------------------
def fig_params():
    def best_by_dev(records, pred):
        cands = [r for r in records if pred(r)]
        return max(cands, key=lambda r: r["dev_f1_mean"])

    scratch = max(RES_A, key=lambda r: r["dev_f1_mean"])
    bert_base = best_by_dev(RES_B, lambda r: r["model_name"] == "bert-base-uncased")
    bert_large = best_by_dev(RES_B, lambda r: r["model_name"] == "bert-large-uncased")
    gpt_base = best_by_dev(RES_B, lambda r: r["model_name"] == "openai-community/gpt2")
    gpt_med = best_by_dev(RES_B, lambda r: r["model_name"] == "openai-community/gpt2-medium")

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.scatter(scratch["n_params"], scratch["slot_f1_mean"], s=150, color=BLUE,
               marker="o", zorder=3)
    ax.scatter([bert_base["n_params"], bert_large["n_params"]],
               [bert_base["slot_f1_mean"], bert_large["slot_f1_mean"]],
               s=150, color=TEAL, marker="^", zorder=3, label="BERT")
    ax.scatter([gpt_base["n_params"], gpt_med["n_params"]],
               [gpt_base["slot_f1_mean"], gpt_med["slot_f1_mean"]],
               s=150, color=ORANGE, marker="s", zorder=3, label="GPT-2")

    ax.set_xscale("log")
    ax.set_xlim(1.3e5, 1.3e9)
    ax.set_ylim(0.86, 0.985)
    ax.set_xlabel("Trainable / total parameters (log scale)")
    ax.set_ylabel("Test Slot F1")
    ax.set_title("Pre-training matters more than scale")
    ax.grid(alpha=0.3, zorder=0)

    # labels: scratch left, BERT above, GPT-2 below -> no overlap
    ax.annotate("GPT-2\nfrom scratch", (scratch["n_params"], scratch["slot_f1_mean"]),
                textcoords="offset points", xytext=(12, -4), ha="left", va="center")
    ax.annotate("BERT-base", (bert_base["n_params"], bert_base["slot_f1_mean"]),
                textcoords="offset points", xytext=(-10, 12), ha="right", va="bottom")
    ax.annotate("BERT-large", (bert_large["n_params"], bert_large["slot_f1_mean"]),
                textcoords="offset points", xytext=(10, 12), ha="left", va="bottom")
    ax.annotate("GPT-2-base", (gpt_base["n_params"], gpt_base["slot_f1_mean"]),
                textcoords="offset points", xytext=(-10, -13), ha="right", va="top")
    ax.annotate("GPT-2-medium", (gpt_med["n_params"], gpt_med["slot_f1_mean"]),
                textcoords="offset points", xytext=(10, -13), ha="left", va="top")
    ax.legend(loc="upper left", framealpha=0.95)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(os.path.join(RT, "nlu_params_vs_f1.png"))
    plt.close(fig)
    print("wrote nlu_params_vs_f1.png")


if __name__ == "__main__":
    fig_perslot()
    fig_confusion()
    fig_params()
