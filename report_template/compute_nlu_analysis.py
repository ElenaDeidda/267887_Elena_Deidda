# Computes test-set error-analysis data for the NLU report:
#   - per-slot-type chunk F1 (CoNLL) for the best BERT and best GPT-2 checkpoints
#   - intent ref/hyp lists (for a confusion matrix) for the best BERT
# Backbones are rebuilt from explicit HF configs (NO pretrained-weight download);
# the full fine-tuned weights come from the saved state_dict in bin/.
import os, sys, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, GPT2Model, GPT2Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
PARTB = os.path.join(ROOT, "NLU", "part_B")
sys.path.insert(0, PARTB)

from utils import (load_data, BERTIntentsAndSlots, GPT2IntentsAndSlots,
                   collate_fn_bert, collate_fn_gpt2,
                   get_bert_tokenizer, get_gpt2_tokenizer, IGNORE_INDEX)
from conll import evaluate

TEST_PATH = os.path.join(PARTB, "dataset", "ATIS", "test.json")
BIN = os.path.join(PARTB, "bin")
OUT = os.path.join(ROOT, "report_template", "nlu_analysis.json")


class LangLite:
    def __init__(self, slot2id, intent2id):
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.id2slot = {v: k for k, v in slot2id.items()}
        self.id2intent = {v: k for k, v in intent2id.items()}


class BERTforNLU(nn.Module):
    def __init__(self, slots_size, n_intents, dropout=0.1):
        super().__init__()
        self.bert = BertModel(BertConfig())  # bert-base-uncased defaults
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(h, slots_size)
        self.intent_out = nn.Linear(h, n_intents)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        last = out.last_hidden_state
        cls = last[:, 0, :]
        return self.slot_out(self.dropout(last)), self.intent_out(self.dropout(cls))


class GPT2forNLU(nn.Module):
    def __init__(self, slots_size, n_intents, dropout=0.1):
        super().__init__()
        # gpt2-medium config (n_embd 1024, 24 layers, 16 heads)
        self.gpt2 = GPT2Model(GPT2Config(n_embd=1024, n_layer=24, n_head=16))
        h = self.gpt2.config.n_embd
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(h, slots_size)
        self.intent_out = nn.Linear(h, n_intents)

    def forward(self, input_ids, attention_mask, seq_lens):
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        cls = torch.stack([last[i, seq_lens[i] - 1] for i in range(last.size(0))])
        return self.slot_out(self.dropout(last)), self.intent_out(self.dropout(cls))


@torch.no_grad()
def run_model(kind, ckpt_path, test_raw):
    ck = torch.load(ckpt_path, map_location="cpu")
    lang = LangLite(ck["slot2id"], ck["intent2id"])
    slots_size, n_intents = len(lang.slot2id), len(lang.intent2id)

    if kind == "bert":
        tok = get_bert_tokenizer("bert-base-uncased")
        ds = BERTIntentsAndSlots(test_raw, lang, tok)
        loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn_bert, shuffle=False)
        model = BERTforNLU(slots_size, n_intents)
    else:
        tok = get_gpt2_tokenizer("openai-community/gpt2")  # same vocab as medium
        ds = GPT2IntentsAndSlots(test_raw, lang, tok)
        loader = DataLoader(ds, batch_size=16, collate_fn=collate_fn_gpt2, shuffle=False)
        model = GPT2forNLU(slots_size, n_intents)

    model.load_state_dict(ck["model"])
    model.eval()

    ref_slots, hyp_slots, ref_int, hyp_int = [], [], [], []
    for bi, batch in enumerate(loader):
        if kind == "bert":
            slots, intent = model(batch["input_ids"], batch["attention_mask"],
                                  batch["token_type_ids"])
        else:
            slots, intent = model(batch["input_ids"], batch["attention_mask"],
                                  batch["seq_lens"])
        slots = slots.permute(0, 2, 1)
        out_int = torch.argmax(intent, dim=1).tolist()
        ref_int += [lang.id2intent[x] for x in batch["intents"].tolist()]
        hyp_int += [lang.id2intent[x] for x in out_int]

        out_slots = torch.argmax(slots, dim=1)
        for s in range(out_slots.size(0)):
            words = batch["words"][s]
            y = batch["y_slots"][s]
            preds = out_slots[s]
            mask = (y != IGNORE_INDEX)
            gt = y[mask].tolist()
            pr = preds[mask].tolist()
            n = len(gt)
            wa = words[:n]
            ref_slots.append([(wa[j], lang.id2slot[gt[j]]) for j in range(n)])
            hyp_slots.append([(wa[j], lang.id2slot[pr[j]]) for j in range(n)])
        print(f"  [{kind}] batch {bi+1}/{len(loader)}", flush=True)

    res = evaluate(ref_slots, hyp_slots)
    acc = sum(int(a == b) for a, b in zip(ref_int, hyp_int)) / len(ref_int)
    perslot = {k: {"f": v["f"], "p": v["p"], "r": v["r"], "s": v["s"]}
               for k, v in res.items()}
    return {"perslot": perslot, "total_f1": res["total"]["f"], "intent_acc": acc,
            "ref_int": ref_int, "hyp_int": hyp_int}


def main():
    test_raw = load_data(TEST_PATH)
    print(f"Test sentences: {len(test_raw)}")
    out = {}
    print("Running BERT...")
    out["bert"] = run_model("bert", os.path.join(BIN, "bert_lr5e-05.pt"), test_raw)
    print(f"  BERT total slot F1 = {out['bert']['total_f1']:.4f} | intent acc = {out['bert']['intent_acc']:.4f}")
    print("Running GPT-2-medium...")
    out["gpt2"] = run_model("gpt2", os.path.join(BIN, "gpt2-medium_lr0.0001_dropout0.1.pt"), test_raw)
    print(f"  GPT-2 total slot F1 = {out['gpt2']['total_f1']:.4f} | intent acc = {out['gpt2']['intent_acc']:.4f}")
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
