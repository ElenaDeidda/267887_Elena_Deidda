# utils.py
# -----------------------------------------------------------------------------
# Data pipeline for Part 1.A: reading the raw Penn TreeBank (PTB) text files,
# wrapping them in a minimal PyTorch Dataset, tokenizing with the pretrained
# GPT-2 BPE tokenizer (used here only as a vocabulary/tokenizer, NOT as a
# pretrained model - the GPT2 model in model.py is trained from scratch), and
# building the train/dev/test DataLoaders used by functions.py and main.py.

import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def read_file(path, eos_token="<eos>"):
    """Read a PTB text file line by line, appending an end-of-sentence token.

    Args:
        path: path to the text file (e.g. dataset/PennTreeBank/ptb.train.txt).
        eos_token: token string marking the end of each sentence/line; lets
            the model learn sentence boundaries during training.

    Returns:
        List of strings (one per line/sentence) with the eos token appended.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(data.Dataset):
    """Minimal PyTorch Dataset wrapping the list of PTB sentences.

    Only stores the raw sentence strings; tokenization is deferred to
    collate_fn so it can run in parallel DataLoader worker processes instead
    of blocking the main process up front.
    """

    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def get_tokenizer(model_name="openai-community/gpt2"):
    """Load the pretrained GPT-2 BPE tokenizer and configure its pad token.

    GPT-2's tokenizer has no native pad token (it was trained for generation,
    not batched fixed-length input), so we reuse the eos token as padding;
    padded positions are excluded from the loss via ignore_index in
    functions.py, so this reuse has no effect on training signal.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def collate_fn(batch, tokenizer):
    """DataLoader collate function: tokenizes a batch and builds LM input/labels.

    Tokenizes the batch of raw sentence strings with padding, then builds the
    input/label pair for language modeling by manually shifting the token
    sequence by one position (input = tokens[:, :-1], label = tokens[:, 1:]),
    so that at each position the model is trained to predict the next token.
    This manual shift is needed because this is a from-scratch model with no
    surrounding library to do it (contrast with Part 1.B, where the
    HuggingFace pretrained model/its trainer handles the shift internally).

    IMPORTANT - stays on CPU on purpose: this function must NOT call
    .to(device) here. With num_workers > 0 (see get_dataloaders), the
    DataLoader runs collate_fn inside separate worker subprocesses, where
    initializing/using CUDA is unsafe (CUDA contexts do not fork safely). The
    actual device transfer happens later, in the main process, inside
    functions.py's train_loop/eval_loop. This also lets the next batch's
    tokenization happen on CPU in parallel while the GPU is busy with the
    current batch.

    Returns:
        input_ids: tensor (B, L) of input token ids.
        labels:    tensor (B, L) of target token ids (input shifted by one).
        n_tokens:  number of non-pad tokens in this batch (Python int), used
            by the caller to compute a token-weighted average loss instead of
            a simple per-batch average.
    """
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    # input = all tokens except the last one
    input_ids = tokenized.input_ids[:, :-1]
    # label = all tokens except the first one -> predict the next token
    labels = tokenized.input_ids[:, 1:]

    # Count non-pad tokens (as a plain int) so the caller can compute a
    # token-weighted average loss across batches/epoch rather than weighting
    # every batch equally regardless of how many real (non-pad) tokens it has.
    n_tokens = int((input_ids != tokenizer.pad_token_id).sum().item())

    return input_ids, labels, n_tokens


def get_dataloaders(train_path, dev_path, test_path, tokenizer, device,
                    train_bs=8, eval_bs=16, num_workers=4):
    """Build and return the train/dev/test DataLoaders.

    Lower train_bs if the GPU runs out of memory. A smaller batch size also
    increases the number of optimizer steps per epoch and acts as a mild
    implicit regularizer (noisier gradient estimates).

    Performance note (not a correctness/results concern): num_workers=4,
    pin_memory and persistent_workers=True are set specifically to remove a
    measured bottleneck - profiling with py-spy showed that roughly half of
    wall-clock training time was spent waiting on synchronous CPU tokenization
    before these were enabled. num_workers>0 parallelizes tokenization
    (collate_fn) across CPU worker processes while the GPU computes on the
    previous batch; pin_memory speeds up the host-to-device copy; and
    persistent_workers avoids respawning worker processes at the start of
    every epoch. None of this affects the model's outputs or final PPL.
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
