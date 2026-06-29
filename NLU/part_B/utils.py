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
#
# ----------------------------------------------------------------------------------
# English summary (module purpose and key rationale)
# ----------------------------------------------------------------------------------
# This module prepares ATIS data for fine-tuning pre-trained BERT and GPT-2 models
# on joint intent classification + slot filling. The central problem it solves is
# sub-word/label alignment: pre-trained tokenizers (WordPiece for BERT, BPE for
# GPT-2) split a single ATIS word into one or more sub-word tokens, but the ATIS
# annotation provides exactly one slot label per WORD, not per sub-token.
#
# The alignment strategy implemented here is "manual per-word sub-tokenization and
# label alignment": each word is tokenized individually
# (tokenizer.tokenize/encode per word), and for every word's resulting sub-token
# sequence, only the FIRST sub-token is assigned the real slot label; every
# subsequent sub-token of that same word, as well as special tokens
# ([CLS]/[SEP] for BERT) and padding, is assigned IGNORE_INDEX (-100), the
# sentinel value CrossEntropyLoss is configured (via ignore_index) to skip when
# computing the slot-filling loss. This is a deliberate, explicit implementation
# choice; it is functionally equivalent to (not better or worse than) the
# alternative HuggingFace approach of calling the tokenizer with
# is_split_into_words=True and post-hoc reconstructing the alignment via
# tokenizer().word_ids() — both achieve the same first-sub-token-carries-label
# semantics, this module simply does it by hand.
#
# Importantly, IGNORE_INDEX (label-loss masking) is independent of attention
# masking: for BERT, attention_mask=1 at [CLS]/[SEP] positions, since the model
# must still attend to them normally as part of the sequence — only their
# CONTRIBUTION TO THE SLOT LOSS is suppressed via IGNORE_INDEX, not their
# visibility to the model.

import json
from collections import Counter

from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PAD_TOKEN = 0       # id del 'pad' in slot2id
IGNORE_INDEX = -100  # standard HuggingFace: ignorato da CrossEntropyLoss
# NOTE on these two constants (easy to conflate, but distinct):
#   - PAD_TOKEN (0) lives in the SLOT-LABEL VOCABULARY (slot2id["pad"] == 0). It
#     is used as a fallback id via slot2id.get(slot, PAD_TOKEN) when a slot string
#     is unexpectedly out of vocabulary. It is a real, valid id that the model CAN
#     predict/be trained on if it ever appears as a target.
#   - IGNORE_INDEX (-100) is the CrossEntropyLoss sentinel that marks a position as
#     EXCLUDED from the slot-filling loss entirely (sub-tokens after the first one
#     in a word, special tokens, and padding positions). It is never a "real" slot
#     class and is never predicted/scored; it just tells the loss "skip this
#     position when computing the average".
# In short: PAD_TOKEN is a slot-vocabulary entry; IGNORE_INDEX is a loss-masking
# signal. They serve unrelated purposes despite both nominally being about "padding".


def load_data(path):
    """Loads a JSON ATIS split (train/dev/test) from disk into a list of dicts."""
    with open(path) as f:
        return json.loads(f.read())


def create_dev_split(train_raw, dev_size=0.10, random_state=42):
    """Dev set stratificato sull'intent; gli intent con un solo esempio restano nel train.

    English: Carves a stratified dev split out of the raw training data,
    stratifying on intent label so that the dev set's intent distribution
    mirrors the train set's. Intents with only a single example cannot be
    stratified (sklearn requires at least 2 members per class to split) and
    are therefore kept entirely in the training set rather than excluded.
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
    """Vocabolari slot2id / intent2id (con gli inversi). slot2id = {'pad':0, slot:1..N};
    le posizioni da ignorare usano -100 (non pad). intent2id non ha 'pad'.

    English: Builds the slot-label and intent-label vocabularies (plus their
    inverses) used to convert between string labels and integer ids. Unlike
    Part 2.A's Lang, this class does NOT build a word2id vocabulary, since
    word-level tokenization is delegated entirely to the pre-trained
    BERT/GPT-2 tokenizer (AutoTokenizer) rather than a custom vocabulary.
    Note that the "pad" entry in slot2id (id 0, PAD_TOKEN) is a distinct
    concept from IGNORE_INDEX (-100): positions to be excluded from the slot
    loss use IGNORE_INDEX, not the "pad" slot class.
    """

    def __init__(self, intents, slots):
        self.slot2id = self._build_slot_vocab(slots)
        self.intent2id = self._build_intent_vocab(intents)
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _build_slot_vocab(self, slots):
        """Builds slot2id with a reserved 'pad' entry at id 0 (PAD_TOKEN),
        followed by the remaining slot labels sorted alphabetically for
        run-to-run reproducibility of the id assignment."""
        vocab = {"pad": PAD_TOKEN}
        for slot in sorted(slots):  # sorted -> riproducibilita'
            if slot not in vocab:
                vocab[slot] = len(vocab)
        return vocab

    def _build_intent_vocab(self, intents):
        """Builds intent2id (no reserved 'pad' entry, since every example
        always has exactly one intent and intents are never padded/ignored)."""
        return {intent: i for i, intent in enumerate(sorted(intents))}


def build_lang(train_raw, dev_raw, test_raw):
    """Etichette slot/intent da train+dev+test.

    English: Collects the full set of slot and intent labels across all
    three splits (train, dev, test) so that Lang's vocabularies cover every
    label that could appear at any stage, then constructs the Lang instance.
    """
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

    English: Wraps ATIS examples for BERT, manually tokenizing each word with
    the WordPiece tokenizer and aligning slot labels sub-token-by-sub-token:
    the first sub-token of a word receives the word's real slot id; any
    additional sub-tokens produced by splitting that same word receive
    IGNORE_INDEX, since the slot label is only meaningful at word
    granularity. [CLS] (prepended) and [SEP] (appended) also receive
    IGNORE_INDEX, as they are not part of any annotated word.
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
                # Manual per-word sub-tokenization: each ATIS word is tokenized
                # independently (rather than tokenizing the whole sentence at
                # once with is_split_into_words=True + word_ids()), which makes
                # the first-sub-token alignment below explicit and direct.
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    word_tokens = [tokenizer.unk_token]
                token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
                # PAD_TOKEN (slot2id["pad"]) is used here only as a defensive
                # fallback for slot strings unexpectedly missing from slot2id;
                # it is unrelated to IGNORE_INDEX below.
                slot_id = lang.slot2id.get(slot, PAD_TOKEN)

                input_ids.extend(token_ids)
                # First sub-token of the word -> real slot id; every other
                # sub-token produced by splitting this same word -> IGNORE_INDEX
                # (excluded from the slot loss, since the label is per-word).
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

    English: Wraps ATIS examples for GPT-2, manually tokenizing each word with
    the BPE tokenizer and aligning slot labels the same way as for BERT (first
    sub-token of a word carries the real label, subsequent sub-tokens get
    IGNORE_INDEX). Unlike BERT, GPT-2 has no built-in [CLS]/[SEP] special
    tokens, so EOS is explicitly appended at the end of the sequence to serve
    as the "[CLS] surrogate" — the position model.py's GPT2forNLU reads the
    intent representation from (see seq_len below). EOS itself gets
    IGNORE_INDEX in the slot labels, since it is not an annotated word.
    """

    def __init__(self, dataset, lang, tokenizer, max_len=128):
        self.samples = []
        for example in dataset:
            words = example["utterance"].split()
            slots = example["slots"].split()
            intent = lang.intent2id[example["intent"]]

            input_ids, slot_labels = [], []
            for word, slot in zip(words, slots):
                # Manual per-word sub-tokenization, mirroring the BERT dataset
                # above: each word is BPE-encoded independently so the
                # first-sub-token alignment is explicit.
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 0:
                    token_ids = [tokenizer.unk_token_id or tokenizer.eos_token_id]
                # PAD_TOKEN fallback for out-of-vocabulary slot strings; distinct
                # from IGNORE_INDEX (see module-level note above).
                slot_id = lang.slot2id.get(slot, PAD_TOKEN)

                input_ids.extend(token_ids)
                # First sub-token -> real slot id; rest -> IGNORE_INDEX.
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
    (tutti 0, frase singola). 'words' resta lista per l'eval conll.

    English: Pads a batch of variable-length BERT examples to the batch's max
    length. input_ids are padded with 0 (BERT's [PAD] id); slot labels are
    padded with IGNORE_INDEX so padding positions never contribute to the
    slot loss; attention_mask is 1 for real tokens (including [CLS]/[SEP],
    which the model must still attend to) and 0 for padding positions only.
    token_type_ids are all 0 since every example is a single segment (no
    sentence-pair task here). 'words' is kept as a plain Python list (not a
    tensor) so the original word strings survive collation for use by the
    conll-style evaluation in functions.py.
    """
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
    attention_mask=0); seq_lens da' la lunghezza reale per estrarre il CLS.

    English: Pads a batch of variable-length GPT-2 examples using the EOS
    token id as the pad value (GPT-2 has no dedicated [PAD] token, so its
    tokenizer's pad_token is set to eos_token in get_gpt2_tokenizer below).
    Because the genuine end-of-sequence EOS and the padding filler share the
    same token id, attention_mask (1 for real tokens including the genuine
    EOS, 0 for padding) is what actually distinguishes them to the model.
    seq_lens records each example's true (un-padded) length, which
    model.py's GPT2forNLU uses to index the correct, per-example last-token
    position for intent classification rather than relying on a fixed
    absolute position in the padded batch.
    """
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
    """Crea i DataLoader con la collate_fn data (BERT o GPT-2).

    English: Wraps the three dataset splits in DataLoaders using whichever
    collate_fn (collate_fn_bert or collate_fn_gpt2) matches the model family.
    Only the train loader shuffles; dev/test preserve order for stable,
    reproducible evaluation.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                              collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval,
                            collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval,
                             collate_fn=collate_fn, shuffle=False)
    return train_loader, dev_loader, test_loader


def get_bert_tokenizer(model_name="bert-base-uncased"):
    """Tokenizer WordPiece di BERT ([CLS]=101, [SEP]=102, [PAD]=0, [UNK]=100).

    English: Loads the pre-trained WordPiece tokenizer matching model_name.
    bert-base-uncased and bert-large-uncased share the same vocabulary, so
    one tokenizer instance is valid for both variants of the BERT family.
    """
    return AutoTokenizer.from_pretrained(model_name)


def get_gpt2_tokenizer(model_name="openai-community/gpt2"):
    """Tokenizer BPE di GPT-2 con pad_token = eos_token (GPT-2 non ha pad/cls/sep nativi).

    English: Loads the pre-trained BPE tokenizer matching model_name and
    explicitly assigns pad_token = eos_token, since GPT-2's tokenizer has no
    native [PAD]/[CLS]/[SEP] special tokens. gpt2 and gpt2-medium share the
    same vocabulary, so one tokenizer instance is valid for both variants of
    the GPT-2 family.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
