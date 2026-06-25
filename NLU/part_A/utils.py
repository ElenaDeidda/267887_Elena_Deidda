# utils.py
# Part 2.A - Caricamento dati per la NLU from scratch su ATIS.
#
# Vocabolario word-level costruito sul training set (niente BPE: ogni parola ha
# esattamente una etichetta slot). Token speciali: pad=0, unk=1, cls=2. Un token
# CLS viene appeso in CODA a ogni frase (modello causale -> solo l'ultima
# posizione vede tutta la frase); il suo target slot condivide l'id 0 col pad,
# quindi e' ignorato dalla loss. Il dev set e' uno split stratificato 10% del train.
#
# Esempio ATIS:
#   {"utterance": "...", "slots": "O B-... ...", "intent": "flight"}

import json
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

PAD_TOKEN = 0  # id del padding


def load_data(path):
    """Carica ATIS da JSON (lista di dict con 'utterance', 'slots', 'intent')."""
    with open(path) as f:
        return json.loads(f.read())


def create_dev_split(train_raw, dev_size=0.10, random_state=42):
    """Dev set stratificato sull'intent (ATIS e' sbilanciato: 'flight' ~70%).
    Gli intent presenti una sola volta restano nel training (non splittabili)."""
    intents = [x["intent"] for x in train_raw]
    count_y = Counter(intents)

    inputs, labels, mini_train = [], [], []
    for example in train_raw:
        if count_y[example["intent"]] > 1:
            inputs.append(example)
            labels.append(example["intent"])
        else:
            mini_train.append(example)

    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=dev_size, random_state=random_state,
        shuffle=True, stratify=labels,
    )
    X_train.extend(mini_train)
    return X_train, X_dev


class Lang:
    """Vocabolari word2id / slot2id / intent2id (con gli inversi id2*).

    Token speciali: 'pad'(0), 'unk'(1), 'cls'(2). In slot2id il 'cls' riceve id=0
    (come pad): cosi' la sua predizione slot e' ignorata dalla loss
    (ignore_index=PAD_TOKEN). id2slot esclude 'cls' per evitare ambiguita'.
    """

    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self._build_word_vocab(words, cutoff=cutoff)
        self.slot2id = self._build_label_vocab(slots, pad=True, cls=True)
        self.intent2id = self._build_label_vocab(intents, pad=False, cls=False)

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items() if k != "cls"}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_word_vocab(self, words, cutoff=0):
        vocab = {"pad": PAD_TOKEN, "unk": 1, "cls": 2}
        for word, freq in Counter(words).items():
            if freq > cutoff and word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _build_label_vocab(self, labels, pad=True, cls=True):
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN  # 0
        for label in sorted(labels):  # sorted -> riproducibilita'
            if label not in vocab:
                vocab[label] = len(vocab)
        if cls:
            vocab["cls"] = PAD_TOKEN   # CLS condivide l'id del pad -> ignorato dalla loss
        return vocab


def build_lang(train_raw, dev_raw, test_raw, cutoff=0):
    """Parole SOLO dal train (le altre -> 'unk' a test time); etichette slot/intent
    da train+dev+test (cosi' nessuna etichetta e' sconosciuta a test time)."""
    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([x["slots"].split() for x in corpus], []))
    intents = set(x["intent"] for x in corpus)
    return Lang(words, intents, slots, cutoff=cutoff)


class IntentsAndSlots(data.Dataset):
    """Dataset ATIS: converte le stringhe in id e appende il CLS in coda.

    Per esempio: utterance = [w1..wN, cls_id], slots = [s1..sN, pad_id]
    (il CLS riceve pad_id -> ignorato dalla loss slot), intent = int.
    """

    def __init__(self, dataset, lang, unk="unk", cls="cls", add_cls=True):
        self.utterances = [x["utterance"] for x in dataset]
        self.slots = [x["slots"] for x in dataset]
        self.intents = [x["intent"] for x in dataset]
        self.unk, self.cls, self.add_cls = unk, cls, add_cls

        self.utt_ids = self._mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self._mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self._mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return {
            "utterance": torch.Tensor(self.utt_ids[idx]),
            "slots": torch.Tensor(self.slot_ids[idx]),
            "intent": self.intent_ids[idx],
        }

    def _mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def _mapping_seq(self, data, mapper):
        result = []
        for seq in data:
            tmp = [mapper[t] if t in mapper else mapper[self.unk] for t in seq.split()]
            if self.add_cls:
                tmp.append(mapper[self.cls])
            result.append(tmp)
        return result


def collate_fn(batch):
    """Padding a destra con PAD_TOKEN. 'slots_len' = lunghezza reale (incluso CLS),
    usata nel forward per estrarre il vettore CLS.

    Returns dict: 'utterances'(B,L), 'y_slots'(B,L), 'intents'(B,), 'slots_len'(B,)."""
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if max(lengths) > 0 else 1
        padded = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded, lengths

    data_by_key = {key: [d[key] for d in batch] for key in batch[0].keys()}
    src_utt, _ = merge(data_by_key["utterance"])
    y_slots, y_lengths = merge(data_by_key["slots"])
    intent = torch.LongTensor(data_by_key["intent"])

    return {
        "utterances": src_utt,
        "intents": intent,
        "y_slots": y_slots,
        "slots_len": torch.LongTensor(y_lengths),
    }


def get_dataloaders(train_dataset, dev_dataset, test_dataset,
                    batch_size_train=128, batch_size_eval=64, device="cpu"):
    """Crea i DataLoader (la collate sposta i tensori su `device`)."""
    def collate_to_device(batch):
        return {k: v.to(device) for k, v in collate_fn(batch).items()}

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                              collate_fn=collate_to_device, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval,
                            collate_fn=collate_to_device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             collate_fn=collate_to_device)
    return train_loader, dev_loader, test_loader
