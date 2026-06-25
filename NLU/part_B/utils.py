# utils.py
# Part 2.B - Dati per il fine-tuning di BERT e GPT-2 su ATIS.
#
# I tokenizer sub-word pre-addestrati spezzano una parola in piu' sub-token, mentre
# ATIS fornisce UNA etichetta slot per parola. Allineamento: il PRIMO sub-token
# riceve l'etichetta reale, gli altri (e i token speciali / il padding) ricevono
# IGNORE_INDEX = -100, che CrossEntropyLoss ignora.
#
#   BERT : WordPiece, [CLS] in posizione 0 (intent), [SEP] in coda.
#   GPT-2: BPE, niente token speciali; EOS appeso in coda come surrogato del CLS,
#          pad = eos. Il padding vero e' distinto dall'EOS via attention_mask = 0.
#
# Lang qui costruisce solo slot2id / intent2id (niente word2id: le parole sono
# gestite dal tokenizer pre-addestrato).

import json
from collections import Counter

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PAD_TOKEN = 0       # id del 'pad' in slot2id
IGNORE_INDEX = -100  # standard HuggingFace: ignorato da CrossEntropyLoss


def load_data(path):
    with open(path) as f:
        return json.loads(f.read())


def create_dev_split(train_raw, dev_size=0.10, random_state=42):
    """Dev set stratificato sull'intent; gli intent con un solo esempio restano nel train."""
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
    """Vocabolari slot2id / intent2id (con gli inversi). slot2id = {'pad':0, slot:1..N};
    le posizioni da ignorare usano -100 (non pad). intent2id non ha 'pad'."""

    def __init__(self, intents, slots):
        self.slot2id = self._build_slot_vocab(slots)
        self.intent2id = self._build_intent_vocab(intents)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_slot_vocab(self, slots):
        vocab = {"pad": PAD_TOKEN}
        for slot in sorted(slots):  # sorted -> riproducibilita'
            if slot not in vocab:
                vocab[slot] = len(vocab)
        return vocab

    def _build_intent_vocab(self, intents):
        return {intent: i for i, intent in enumerate(sorted(intents))}


def build_lang(train_raw, dev_raw, test_raw):
    """Etichette slot/intent da train+dev+test."""
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([x["slots"].split() for x in corpus], []))
    intents = set(x["intent"] for x in corpus)
    return Lang(intents, slots)


# ----------------------------------------------------------------------------
# Dataset per BERT (WordPiece)
# ----------------------------------------------------------------------------

class BERTIntentsAndSlots(data.Dataset):
    """ATIS con WordPiece e allineamento slot.

    Esempio: [CLS] fly ##ing from New York [SEP]
             -100  O  -100  O    B   I     -100
    [CLS], [SEP] e i sub-token dopo il primo ricevono -100.
    """

    def __init__(self, dataset, lang, tokenizer, max_len=128):
        self.samples = []
        for example in dataset:
            words = example["utterance"].split()
            slots = example["slots"].split()
            intent = lang.intent2id[example["intent"]]

            input_ids = [tokenizer.cls_token_id]
            slot_labels = [IGNORE_INDEX]  # [CLS] -> -100

            for word, slot in zip(words, slots):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    word_tokens = [tokenizer.unk_token]
                token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
                slot_id = lang.slot2id.get(slot, PAD_TOKEN)

                input_ids.extend(token_ids)
                # primo sub-token -> etichetta reale; gli altri -> -100
                slot_labels.extend([slot_id] + [IGNORE_INDEX] * (len(token_ids) - 1))

            input_ids.append(tokenizer.sep_token_id)  # [SEP] in coda
            slot_labels.append(IGNORE_INDEX)

            if len(input_ids) > max_len:  # truncation (raro su ATIS)
                input_ids = input_ids[:max_len - 1] + [tokenizer.sep_token_id]
                slot_labels = slot_labels[:max_len - 1] + [IGNORE_INDEX]

            self.samples.append({"input_ids": input_ids, "slot_labels": slot_labels,
                                 "intent": intent, "words": words})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
                "slot_labels": torch.tensor(s["slot_labels"], dtype=torch.long),
                "intent": s["intent"], "words": s["words"]}


# ----------------------------------------------------------------------------
# Dataset per GPT-2 (BPE)
# ----------------------------------------------------------------------------

class GPT2IntentsAndSlots(data.Dataset):
    """ATIS con BPE (GPT-2) e allineamento slot.

    Esempio: fly ing from New York <eos>
             O  -100  O   B   I    -100
    GPT-2 non ha token speciali automatici: appendiamo EOS in coda come surrogato
    del CLS per l'intent (riceve -100 negli slot).
    """

    def __init__(self, dataset, lang, tokenizer, max_len=128):
        self.samples = []
        for example in dataset:
            words = example["utterance"].split()
            slots = example["slots"].split()
            intent = lang.intent2id[example["intent"]]

            input_ids, slot_labels = [], []
            for word, slot in zip(words, slots):
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 0:
                    token_ids = [tokenizer.unk_token_id or tokenizer.eos_token_id]
                slot_id = lang.slot2id.get(slot, PAD_TOKEN)

                input_ids.extend(token_ids)
                slot_labels.extend([slot_id] + [IGNORE_INDEX] * (len(token_ids) - 1))

            input_ids.append(tokenizer.eos_token_id)  # EOS = surrogato del CLS
            slot_labels.append(IGNORE_INDEX)

            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len - 1] + [tokenizer.eos_token_id]
                slot_labels = slot_labels[:max_len - 1] + [IGNORE_INDEX]

            self.samples.append({"input_ids": input_ids, "slot_labels": slot_labels,
                                 "intent": intent, "words": words,
                                 "seq_len": len(input_ids)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {"input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
                "slot_labels": torch.tensor(s["slot_labels"], dtype=torch.long),
                "intent": s["intent"], "words": s["words"], "seq_len": s["seq_len"]}


# ----------------------------------------------------------------------------
# Collate functions
# ----------------------------------------------------------------------------

def collate_fn_bert(batch):
    """Padding input_ids con 0, y_slots con -100, + attention_mask e token_type_ids
    (tutti 0, frase singola). 'words' resta lista per l'eval conll."""
    input_ids_list = [item["input_ids"] for item in batch]
    slot_labels_list = [item["slot_labels"] for item in batch]
    intents = torch.tensor([item["intent"] for item in batch], dtype=torch.long)
    words_list = [item["words"] for item in batch]

    max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids, padded_slot_labels, attention_masks = [], [], []
    for input_ids, slot_labels in zip(input_ids_list, slot_labels_list):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_slot_labels.append(torch.cat([slot_labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(input_ids), dtype=torch.long),
                                          torch.zeros(pad_len, dtype=torch.long)]))

    return {"input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "token_type_ids": torch.zeros(len(batch), max_len, dtype=torch.long),
            "y_slots": torch.stack(padded_slot_labels),
            "intents": intents, "words": words_list}


def collate_fn_gpt2(batch):
    """Padding con l'id EOS (il padding e' distinto dal vero EOS/CLS via
    attention_mask=0); seq_lens da' la lunghezza reale per estrarre il CLS."""
    input_ids_list = [item["input_ids"] for item in batch]
    slot_labels_list = [item["slot_labels"] for item in batch]
    intents = torch.tensor([item["intent"] for item in batch], dtype=torch.long)
    seq_lens = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)
    words_list = [item["words"] for item in batch]

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = input_ids_list[0][-1].item()  # EOS: ultimo token di ogni sequenza

    padded_input_ids, padded_slot_labels, attention_masks = [], [], []
    for input_ids, slot_labels in zip(input_ids_list, slot_labels_list):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        padded_slot_labels.append(torch.cat([slot_labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(input_ids), dtype=torch.long),
                                          torch.zeros(pad_len, dtype=torch.long)]))

    return {"input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "y_slots": torch.stack(padded_slot_labels),
            "intents": intents, "seq_lens": seq_lens, "words": words_list}


def get_dataloaders(train_dataset, dev_dataset, test_dataset, collate_fn,
                    batch_size_train=32, batch_size_eval=64):
    """Crea i DataLoader con la collate_fn data (BERT o GPT-2)."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                              collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval,
                            collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             collate_fn=collate_fn, shuffle=False)
    return train_loader, dev_loader, test_loader


def get_bert_tokenizer(model_name="bert-base-uncased"):
    """Tokenizer WordPiece di BERT ([CLS]=101, [SEP]=102, [PAD]=0, [UNK]=100)."""
    return AutoTokenizer.from_pretrained(model_name)


def get_gpt2_tokenizer(model_name="openai-community/gpt2"):
    """Tokenizer BPE di GPT-2 con pad_token = eos_token (GPT-2 non ha pad/cls/sep nativi)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
