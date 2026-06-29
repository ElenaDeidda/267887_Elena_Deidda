# utils.py
# Part 1.B - Data loading for LoRA fine-tuning of GPT-2 on the Penn TreeBank corpus.
#
# Structurally similar to Part 1.A's utils.py: GPT-2's BPE tokenizer (pad token aliased
# to eos token, since GPT-2 ships with no dedicated pad token) and per-batch dynamic
# padding inside collate_fn. collate_fn still returns the triple
# (input_ids, labels, n_tokens) to keep a common interface with Part 1.A's loaders, but
# in this part the `labels` tensor produced here is NOT the one actually used for the
# loss: the training/eval loops in functions.py discard it and build their own labels
# from input_ids (with padding positions set to -100), because GPT2LMHeadModel performs
# the input/label shift internally when given `labels=...`. Re-shifting here as well
# would double-shift and silently corrupt the loss, so this module deliberately leaves
# that responsibility to functions.py.
#
# DataLoader note: batch_size_train defaults to 8 here (not bumped to 32 as in the fixed
# Part 1.A) -- kept at the original default through the whole 3-step greedy search for
# consistency, since switching batch size mid-search would make later steps (alpha)
# incomparable to earlier ones (rank) which were run at batch=8. Likewise num_workers
# defaults to 0 (no parallel data loading workers) and pin_memory is not enabled; this
# is the same pattern Part 1.A had before being fixed, left as-is here deliberately for
# search consistency -- it is not a bug to fix, just a documented characteristic.

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from functools import partial


def read_file(path, eos_token="<eos>"):
    """Read a corpus file line by line and append an end-of-sequence token to each
    sentence (PTB ships pre-tokenized, one sentence per line, without an explicit EOS
    marker)."""
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(Dataset):
    """Thin Dataset wrapper around a list of raw sentence strings.

    Tokenization is deferred to collate_fn (via the DataLoader) rather than done eagerly
    here, so that padding can be computed per-batch instead of globally.
    """

    def __init__(self, corpus):
        self.sents = list(corpus)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def get_tokenizer(model_name="openai-community/gpt2"):
    """Load GPT-2's BPE tokenizer and alias pad_token to eos_token.

    GPT-2 was pre-trained without a dedicated padding token, so the standard
    workaround is to reuse eos_token for padding. Padding positions are later replaced
    with -100 in the training/eval loops (functions.py) so HuggingFace's internal loss
    computation ignores them rather than being confused by the reused eos/pad id.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def collate_fn(batch, tokenizer, device):
    """Tokenize and pad one batch of raw sentence strings.

    Returns:
        input_ids : (B, L-1) tensor, token[0..L-2] of each padded sequence.
        labels    : (B, L-1) tensor, token[1..L-1] -- kept only for interface
                    compatibility with Part 1.A's loaders; in Part 1.B this is NOT
                    the tensor actually used to compute the loss (see module docstring
                    above and functions.py: GPT2LMHeadModel does its own internal
                    shift when given labels=input_ids, so using this pre-shifted
                    tensor as well would shift twice).
        n_tokens  : count of non-padding tokens in input_ids, used to compute a
                    correctly token-weighted average loss across batches of unequal
                    length.
    """
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    n_tokens = int((input_ids != tokenizer.pad_token_id).sum().item())

    return input_ids, labels, n_tokens


def get_dataloaders(train_raw, dev_raw, test_raw, tokenizer, device,
                    batch_size_train=8, batch_size_eval=16):
    """Build the train/dev/test DataLoaders (shuffling enabled only for training).

    Uses default DataLoader settings (no parallel workers, no pin_memory) -- see the
    module-level note above on why this was deliberately left unchanged for this part.
    """
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
