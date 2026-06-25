# utils.py
# Part 1.B - Caricamento dati per il fine-tuning LoRA di GPT-2 sul Penn TreeBank.
#
# Praticamente identico alla Part 1.A: tokenizer BPE di GPT-2 (pad = eos) e
# padding per-batch nella collate_fn. La collate_fn restituisce comunque
# (input_ids, labels, n_tokens) per avere un'interfaccia comune, ma in 1.B il
# loop di training ignora `labels` e lascia che GPT2LMHeadModel faccia lo shift
# internamente (vedi functions.py).

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from functools import partial


def read_file(path, eos_token="<eos>"):
    """Legge il file riga per riga e appende il token di fine sequenza a ogni frase."""
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(Dataset):
    """Dataset di frasi raw (stringhe); la tokenizzazione avviene nella collate_fn."""

    def __init__(self, corpus):
        self.sents = list(corpus)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def get_tokenizer(model_name="openai-community/gpt2"):
    """Tokenizer BPE di GPT-2 con pad_token = eos_token.

    Le posizioni di padding verranno poi rimpiazzate con -100 nel loop di
    training, cosi' la loss di HuggingFace le ignora.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def collate_fn(batch, tokenizer, device):
    """Tokenizza e padda un batch.

    Returns:
        input_ids : (B, L-1) - token[0..L-2]
        labels    : (B, L-1) - token[1..L-1] (usati solo come interfaccia; in 1.B ignorati)
        n_tokens  : numero di token non-pad (per la media pesata della loss)
    """
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    n_tokens = int((input_ids != tokenizer.pad_token_id).sum().item())

    return input_ids, labels, n_tokens


def get_dataloaders(train_raw, dev_raw, test_raw, tokenizer, device,
                    batch_size_train=8, batch_size_eval=16):
    """Crea i tre DataLoader (shuffle solo sul training)."""
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    collate = partial(collate_fn, tokenizer=tokenizer, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                              shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval,
                            shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             shuffle=False, collate_fn=collate)

    return train_loader, dev_loader, test_loader
