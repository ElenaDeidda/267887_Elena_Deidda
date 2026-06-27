# utils.py
# Tutte le funzioni per il pre-processing e il caricamento del dataset Penn Treebank (PTB).
# Qui costruiamo: lettura del corpus, Dataset PyTorch, tokenizer di GPT-2 e i DataLoader.

import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def read_file(path, eos_token="<eos>"):
    """Legge il file riga per riga e aggiunge il token di fine sequenza a ogni frase.

    Args:
        path: percorso del file di testo (es. dataset/PennTreeBank/ptb.train.txt)
        eos_token: token che segna la fine di una frase

    Returns:
        Lista di stringhe (una per frase) con l'eos in coda.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(data.Dataset):
    """Dataset PyTorch minimale: incapsula la lista di frasi del corpus.

    I metodi obbligatori sono __init__, __len__ e __getitem__.
    """

    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def get_tokenizer(model_name="openai-community/gpt2"):
    """Carica il tokenizer (BPE) di GPT-2 e imposta il pad token.

    GPT-2 non ha un pad token nativo: usiamo l'eos token come padding.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def collate_fn(batch, tokenizer):
    """Funzione di collate per il DataLoader.

    Tokenizza il batch con padding, poi costruisce input e label per il
    Language Modeling: la label e' la sequenza di input shiftata a sinistra
    di una posizione (per ogni token si predice il token successivo).

    Resta su CPU: con num_workers>0 questa funzione gira in processi worker,
    dove inizializzare CUDA non e' sicuro. Lo spostamento su device avviene
    nel training/eval loop (functions.py), cosi' il prossimo batch puo' essere
    tokenizzato in parallelo mentre la GPU lavora su quello corrente.

    Returns:
        input_ids: tensore (B, L) dei token di input
        labels:    tensore (B, L) dei token target (input shiftato)
        n_tokens:  numero di token non-pad nel batch (per la media della loss)
    """
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    # input = tutti i token tranne l'ultimo
    input_ids = tokenized.input_ids[:, :-1]
    # label = tutti i token tranne il primo -> predici il token successivo
    labels = tokenized.input_ids[:, 1:]

    # contiamo i token non-pad (come intero, per usarlo nelle medie pesate)
    n_tokens = int((input_ids != tokenizer.pad_token_id).sum().item())

    return input_ids, labels, n_tokens


def get_dataloaders(train_path, dev_path, test_path, tokenizer, device,
                    train_bs=8, eval_bs=16, num_workers=4):
    """Costruisce e restituisce i tre DataLoader (train, dev, test).

    Riduci train_bs se la GPU non ha abbastanza memoria. Un batch piu' piccolo
    aumenta gli step di backpropagation e funge da leggera regolarizzazione.

    num_workers>0 parallelizza la tokenizzazione su CPU rispetto al calcolo
    GPU; pin_memory + persistent_workers riducono ulteriormente l'overhead
    per batch quando si allena su GPU.
    """
    train_raw = read_file(train_path)
    dev_raw = read_file(dev_path)
    test_raw = read_file(test_path)

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    cf = partial(collate_fn, tokenizer=tokenizer)
    pin = device != "cpu"

    train_loader = DataLoader(train_dataset, batch_size=train_bs,
                              collate_fn=cf, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_bs, collate_fn=cf,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, collate_fn=cf,
                             num_workers=num_workers, pin_memory=pin,
                             persistent_workers=num_workers > 0)

    return train_loader, dev_loader, test_loader
