# utils.py
# Part 2.A - Data loading and preprocessing for the from-scratch NLU model on ATIS.
#
# Word-level vocabulary built from the training set only (no BPE/subword
# tokenization: each whitespace-separated word maps to exactly one slot
# label, matching ATIS's pre-tokenized BIO annotation). Special tokens:
# pad=0, unk=1, cls=2.
#
# A CLS token is appended at the END of every utterance (not at the
# beginning, unlike BERT-style encoders). This is required because the
# downstream model (model.py) is autoregressive/causal: a token can only
# attend to earlier positions, so only the LAST position in the sequence has
# attended to the entire utterance. Placing CLS at the end makes its hidden
# state the correct pooled representation for intent classification. Its
# slot target shares id 0 with PAD (see Lang._build_label_vocab below), so it
# is automatically excluded from the slot loss via ignore_index=PAD_TOKEN.
#
# The dev set is a stratified 10% split of the original training set.
#
# ATIS example record:
#   {"utterance": "...", "slots": "O B-... ...", "intent": "flight"}

import json
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

PAD_TOKEN = 0  # padding id, shared with the CLS slot-label id (see Lang)


def load_data(path):
    """Load an ATIS split from a JSON file (a list of dicts, each with
    'utterance', 'slots', and 'intent' string fields)."""
    with open(path) as f:
        return json.loads(f.read())


def create_dev_split(train_raw, dev_size=0.10, random_state=42):
    """Carve out a stratified-by-intent dev split from the training set.

    ATIS's intent distribution is heavily skewed (the 'flight' intent alone
    covers roughly 70% of examples), so stratified sampling is used to keep
    the dev set's intent proportions representative of the training set.
    Intents that occur only once cannot be stratified (sklearn requires at
    least 2 members per class to split) and are therefore always kept in the
    training portion rather than risking an error or being dropped.
    """
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
    """Holds the word2id / slot2id / intent2id vocabularies (and their id2*
    inverses) used to convert raw ATIS strings to integer ids.

    Special tokens: 'pad' (0), 'unk' (1), 'cls' (2) in word2id. In slot2id,
    'cls' is deliberately mapped to id 0, the SAME id as 'pad': this means
    the slot-loss's ignore_index=PAD_TOKEN (see functions.py) automatically
    ignores the CLS position's slot prediction at training/eval time, without
    needing a separate special-case. id2slot excludes the 'cls' key (since it
    aliases id 0 with 'pad', keeping both would be ambiguous when decoding
    ids back to slot label strings).
    """

    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self._build_word_vocab(words, cutoff=cutoff)
        self.slot2id = self._build_label_vocab(slots, pad=True, cls=True)
        self.intent2id = self._build_label_vocab(intents, pad=False, cls=False)

        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items() if k != "cls"}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_word_vocab(self, words, cutoff=0):
        """Build word2id from a flat list of training-set words, keeping only
        words with frequency > cutoff (cutoff=0 keeps everything)."""
        vocab = {"pad": PAD_TOKEN, "unk": 1, "cls": 2}
        for word, freq in Counter(words).items():
            if freq > cutoff and word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _build_label_vocab(self, labels, pad=True, cls=True):
        """Build a label vocabulary (used for both slot2id and intent2id).
        Labels are sorted before assigning ids purely for reproducibility
        (so the same corpus always yields the same id assignment)."""
        vocab = {}
        if pad:
            vocab["pad"] = PAD_TOKEN  # 0
        for label in sorted(labels):  # sorted -> reproducible id assignment
            if label not in vocab:
                vocab[label] = len(vocab)
        if cls:
            vocab["cls"] = PAD_TOKEN   # CLS aliases the pad id -> ignored by the slot loss
        return vocab


def build_lang(train_raw, dev_raw, test_raw, cutoff=0):
    """Build the Lang vocabularies.

    Word vocabulary is built ONLY from the training set (any word seen only
    at dev/test time falls back to 'unk'), matching standard practice and
    preventing test-set leakage into the vocabulary. Slot and intent label
    vocabularies, by contrast, are built from train+dev+test combined, so
    that no label encountered at evaluation time is ever truly "unseen" by
    the model's output space (labels are a small closed set, unlike words).
    """
    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([x["slots"].split() for x in corpus], []))
    intents = set(x["intent"] for x in corpus)
    return Lang(words, intents, slots, cutoff=cutoff)


class IntentsAndSlots(data.Dataset):
    """PyTorch Dataset wrapping ATIS examples, converting word/slot/intent
    strings to integer ids and appending the CLS token id at the end of each
    utterance and slot sequence.

    For one example: utterance ids = [w1..wN, cls_id], slot ids = [s1..sN,
    pad_id] (the CLS position is assigned pad_id, so it is ignored by the
    slot loss), intent id = a single int.
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
        """Map a list of (intent) label strings to ids, defaulting to 'unk'
        for any unseen label."""
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def _mapping_seq(self, data, mapper):
        """Map a list of whitespace-tokenized sequences (utterances or BIO
        slot strings) to lists of ids, appending the CLS id at the end of
        each sequence when add_cls=True. The CLS append happens here, at the
        per-sequence level, so every example handed to the model already has
        CLS in the final position before any batching/padding occurs."""
        result = []
        for seq in data:
            tmp = [mapper[t] if t in mapper else mapper[self.unk] for t in seq.split()]
            if self.add_cls:
                tmp.append(mapper[self.cls])
            result.append(tmp)
        return result


def collate_fn(batch):
    """Collate a list of dataset examples into a padded batch.

    Right-pads variable-length sequences with PAD_TOKEN up to the batch's max
    length. 'slots_len' records each example's true (unpadded) length,
    INCLUDING the appended CLS token; this is exactly the value the model's
    forward() needs to index out each example's own CLS hidden state at
    position slots_len[i]-1 (see model.py), since the batch's padded length
    may exceed any individual example's real length.

    Returns dict: 'utterances' (B, L), 'y_slots' (B, L), 'intents' (B,),
    'slots_len' (B,).
    """
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
    """Build train/dev/test DataLoaders. The collate function used here wraps
    collate_fn and additionally moves every resulting tensor to `device`, so
    batches arrive on the target device (CPU/CUDA/MPS) ready for the model."""
    def collate_to_device(batch):
        return {k: v.to(device) for k, v in collate_fn(batch).items()}

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                              collate_fn=collate_to_device, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval,
                            collate_fn=collate_to_device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             collate_fn=collate_to_device)
    return train_loader, dev_loader, test_loader
