# model.py
# Part 2.B - Fine-tuning di BERT (encoder) e GPT-2 (decoder) pre-addestrati per
# il joint intent classification + slot filling.
#
# Entrambi condividono la stessa struttura di teste sopra un backbone pre-addestrato:
#   dropout -> slot_out  : Linear(hidden, slots_size)  (una etichetta per token)
#           -> intent_out: Linear(hidden, n_intents)   (una etichetta per frase)
#
# Vettore per l'intent: BERT usa il [CLS] in posizione 0 (e' bidirezionale);
# GPT-2 e' causale, quindi usa il token EOS appeso in CODA (l'unica posizione che
# ha visto tutta la frase). hidden_size e' letto dalla config (768 base /
# 1024 large/medium). Fine-tuning completo: tutti i pesi del backbone sono addestrabili.
#
# ----------------------------------------------------------------------------------
# English summary (module purpose and key rationale)
# ----------------------------------------------------------------------------------
# This module defines the two joint intent-classification + slot-filling heads used
# in Part 2.B, one wrapping a pre-trained BERT encoder and one wrapping a pre-trained
# GPT-2 decoder (`AutoModel.from_pretrained`, full fine-tuning: every backbone weight
# is trainable, not frozen). Both heads share the same architecture on top of the
# backbone's last hidden states:
#   - a per-token slot-filling head:   Linear(hidden_size, slots_size)
#   - a per-sequence intent head:      Linear(hidden_size, n_intents)
# applied after a shared nn.Dropout.
#
# The key architectural asymmetry between the two classes is WHICH hidden state is
# fed into the intent head:
#   - BERTforNLU uses the hidden state at position 0, i.e. the [CLS] token. This is
#     standard practice for BERT because BERT is bidirectional/self-attention over
#     the full sequence at every layer, so every position (including [CLS]) already
#     attends to the entire utterance. [CLS] is the conventional aggregation point
#     BERT itself was pre-trained to use for sequence-level tasks.
#   - GPT2forNLU CANNOT do the same: GPT-2 is causal (unidirectional, left-to-right
#     attention only), so position 0 has only seen itself, not the rest of the
#     sentence. Only the LAST real (non-padding) token of each sequence has attended
#     to the whole utterance. GPT2forNLU therefore extracts the hidden state at the
#     last real token position, individually per example via `seq_lens` (the true,
#     un-padded sequence length), rather than at a fixed absolute position in the
#     padded batch. This mirrors the same constraint encountered with the
#     from-scratch GPT-2 in Part 2.A, but here applied to a pre-trained GPT-2 model.

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTforNLU(nn.Module):
    """BERT fine-tuned per intent + slot. Intent dal [CLS] (posizione 0, che BERT
    pre-addestra per aggregare l'intera sequenza); slot da ogni token.

    English: Joint intent-classification + slot-filling head on top of a
    pre-trained BERT encoder (e.g. bert-base-uncased / bert-large-uncased).
    Because BERT attends bidirectionally over the full sequence at every layer,
    the hidden state at position 0 ([CLS]) already encodes the whole utterance,
    so it is used directly as the sentence-level representation for intent
    classification. Every backbone parameter is fine-tuned (not frozen).

    Args:
        slots_size (int): size of the slot label vocabulary (output dim of slot_out).
        n_intents (int): number of intent classes (output dim of intent_out).
        model_name (str): HuggingFace model id passed to AutoModel.from_pretrained
            (e.g. "bert-base-uncased" or "bert-large-uncased").
        dropout (float): dropout probability applied before each linear head.
    """

    def __init__(self, slots_size, n_intents,
                 model_name="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 base / 1024 large

        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Returns slots (B, L, slots_size), intent (B, n_intents).

        Note: attention_mask is 1 at [CLS]/[SEP] positions too (BERT still
        attends to them normally as part of the sequence) even though their
        slot label is later set to IGNORE_INDEX during loss computation in
        utils.py/functions.py — attention masking and loss masking are
        independent concerns.
        """
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        last_hidden = output.last_hidden_state  # (B, L, hidden_size)

        # Sentence-level representation for intent classification: the [CLS]
        # token at position 0. Valid here specifically because BERT is
        # bidirectional, so [CLS] has already attended to (and aggregated)
        # every other token in the sequence.
        cls_repr = last_hidden[:, 0, :]  # [CLS] in posizione 0

        slots = self.slot_out(self.dropout(last_hidden))  # (B, L, slots_size)
        intent = self.intent_out(self.dropout(cls_repr))  # (B, n_intents)
        return slots, intent


class GPT2forNLU(nn.Module):
    """GPT-2 fine-tuned per intent + slot. GPT-2 e' causale: l'intent si estrae dal
    token EOS appeso in CODA (l'unica posizione che ha visto tutta la frase); slot
    da ogni token.

    English: Joint intent-classification + slot-filling head on top of a
    pre-trained GPT-2 decoder (e.g. openai-community/gpt2 / gpt2-medium).
    Unlike BERT, GPT-2's self-attention is causal (each position can only
    attend to itself and earlier positions), so no early position has seen
    the full utterance. The EOS token appended at the END of each input
    sequence (see utils.py) is the only position whose hidden state has
    attended to every other token, so it is used as the sentence-level
    representation for intent classification. Every backbone parameter is
    fine-tuned (not frozen).

    Args:
        slots_size (int): size of the slot label vocabulary (output dim of slot_out).
        n_intents (int): number of intent classes (output dim of intent_out).
        model_name (str): HuggingFace model id passed to AutoModel.from_pretrained
            (e.g. "openai-community/gpt2" or "openai-community/gpt2-medium").
        dropout (float): dropout probability applied before each linear head.
    """

    def __init__(self, slots_size, n_intents,
                 model_name="openai-community/gpt2", dropout=0.1):
        super().__init__()
        self.gpt2 = AutoModel.from_pretrained(model_name)
        hidden_size = self.gpt2.config.n_embd  # 768 gpt2 / 1024 gpt2-medium

        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, seq_lens):
        """seq_lens: (B,) lunghezza reale (incluso EOS). Returns slots, intent.

        seq_lens carries the true, un-padded sequence length of each example
        in the batch (1-indexed, i.e. counting from 1, so the last real token
        sits at index seq_lens[i] - 1). It is required precisely because GPT-2
        has no [CLS]/[SEP] special tokens and the batch is right-padded to the
        longest sequence: the last ABSOLUTE position in the padded batch is
        usually a padding token, not the real last token, so we must index
        per-example using seq_lens rather than a fixed position like -1.
        """
        output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state  # (B, L, hidden_size)

        # Sentence-level representation for intent classification: the
        # EOS/"CLS-surrogate" hidden state at each example's true last
        # position (seq_lens[i] - 1), NOT at a fixed absolute position in the
        # padded batch. This is the GPT-2-specific counterpart of BERT's
        # last_hidden[:, 0, :]: because attention is causal, only the last
        # real token has attended to the whole utterance, and that position
        # differs per example once the batch is padded.
        cls_repr = torch.stack([
            last_hidden[i, seq_lens[i] - 1] for i in range(last_hidden.size(0))
        ])  # (B, hidden_size)

        slots = self.slot_out(self.dropout(last_hidden))  # (B, L, slots_size)
        intent = self.intent_out(self.dropout(cls_repr))  # (B, n_intents)
        return slots, intent


def count_parameters(model):
    """Numero di parametri addestrabili (utile per confrontare le varianti).

    English: Counts only parameters with requires_grad=True, i.e. trainable
    parameters. Since fine-tuning here is full (the entire backbone is
    unfrozen), this effectively reports the model's total trainable size —
    used to compare base vs. large/medium variants (the experiment shows
    ~3x more parameters did not translate into better test metrics).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
