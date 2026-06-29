# model.py
# Part 2.A - GPT-2 from scratch for joint Intent Classification + Slot Filling.
#
# Same backbone as Part 1.A (causal multi-head attention, feed-forward network,
# Pre-LN transformer blocks, learned token + positional embeddings, causal
# mask) but with two output heads instead of the language-modeling head:
#   slot_out   : Linear(d_model, slots_size) - one label per token (sequence labeling)
#   intent_out : Linear(d_model, n_intents)  - one label per utterance, read off the CLS token
#
# IMPORTANT DESIGN CHOICE - CLS placed at the END of the sequence, not the start:
# this model is autoregressive/causal, so a token can only attend to tokens at
# or before its own position. A BERT-style [CLS] prepended at the START would
# only ever see itself (nothing follows it for attention purposes... actually
# it would see everything because it's first - but it gets attended-to without
# itself attending forward), and would not carry information accumulated from
# the rest of the sentence. By appending CLS at the END instead, it is the
# unique position that has attended to the ENTIRE utterance, making its final
# hidden state the only sensible representation to pool for intent
# classification. Its target slot label is PAD (id 0), so it is excluded from
# the slot loss via ignore_index (see functions.py).
#
# Incremental additions specific to this assignment (2.A) compared to the
# Part 1.A backbone:
#   - hyperparameter search over d_model, n_heads, num_layers, ff_dim (driven
#     from main.py, not hardcoded here)
#   - dropout (controlled by the `dropout` argument), including a dropout
#     layer applied right before the two output heads.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention (same mechanism as in Part 1.A).

    Splits the model dimension into `n_heads` independent attention heads,
    computes scaled dot-product attention with an additive causal mask (so
    position i cannot attend to positions j > i), then merges the heads back
    and projects to d_model.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) deve essere divisibile per n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """Reshape (B, L, d_model) -> (B, n_heads, L, d_k) so attention can be
        computed independently per head via batched matmuls."""
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)

    def _merge_heads(self, x):
        """Inverse of _split_heads: (B, n_heads, L, d_k) -> (B, L, d_model)."""
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, x, mask=None):
        """x: (B, L, d_model) input hidden states.
        mask: additive causal mask broadcastable to (B, n_heads, L, L), with
        -inf on disallowed (future) positions and 0 elsewhere; added to the
        raw attention scores before softmax so future positions get ~0
        probability.
        Returns: (B, L, d_model) attended and re-projected output.
        """
        Q = self._split_heads(self.w_q(x))
        K = self._split_heads(self.w_k(x))
        V = self._split_heads(self.w_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:        # additive mask: -inf on future positions (causal)
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)                 # (B, n_heads, L, d_k)
        out = self.out_proj(self._merge_heads(context))
        return self.proj_dropout(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear -> Dropout.

    Applied independently to each position/token after attention, following
    the standard Transformer block design.
    """

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: x = x + Attn(LN(x)); x = x + FF(LN(x)).

    Pre-LayerNorm (normalize before the sub-layer, residual add after) is the
    GPT-2-style variant, which tends to be more stable to train than the
    original Post-LN ("Attention is All You Need") ordering.
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """GPT-2-style decoder-only Transformer trained from scratch, with two
    output heads for joint slot filling (per-token) and intent classification
    (per-utterance, read from the trailing CLS token).

    Args:
        vocab_size: size of the word-level vocabulary (includes pad/unk/cls).
        slots_size: number of slot labels (BIO tags), includes the pad/cls id.
        n_intents: number of distinct intent classes.
        pos_emb_size: maximum sequence length supported by the learned
            positional embedding table (also the size of the causal mask
            buffer); asserted against at forward time.
        d_model, n_heads, num_layers, ff_dim, dropout: standard Transformer
            sizing hyperparameters, swept over in main.py's greedy search.
    """

    def __init__(
        self,
        vocab_size,
        slots_size,
        n_intents,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.0,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        # Token embedding (covers special ids pad=0, unk=1, cls=2) + learned
        # positional embedding (as in GPT-2, not sinusoidal/fixed).
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Addition specific to this assignment (2.A): a dropout layer right
        # before the two output heads, on top of the per-block dropout
        # inherited from the Part 1.A backbone.
        self.out_dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(d_model, slots_size)     # one label per token (sequence labeling)
        self.intent_out = nn.Linear(d_model, n_intents)    # one label per utterance, from CLS

        # Causal mask precomputed once and registered as a (non-trainable)
        # buffer: row i can only attend to columns j <= i (upper triangle,
        # excluding the diagonal, is set to -inf). Because CLS is appended at
        # the END of each sequence, it occupies the last valid row and is
        # therefore the only position whose attention spans the whole
        # utterance — which is exactly why it is suitable as the pooled
        # representation for intent classification.
        mask = torch.triu(torch.full((pos_emb_size, pos_emb_size), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        """Run the model on a padded batch of token-id sequences.

        Args:
            idx: (B, L) LongTensor of token ids, right-padded with PAD_TOKEN,
                with the CLS token appended at the end of each real sequence
                (so for a given example, valid content occupies indices
                [0, seq_lens[i]-1], where index seq_lens[i]-1 is CLS itself).
            seq_lens: (B,) LongTensor, the real (unpadded) length of each
                sequence INCLUDING the appended CLS token.

        Returns:
            slots: (B, L, slots_size) per-token slot logits (the logits at
                padding positions and at the CLS position are produced but are
                meaningless / ignored downstream via ignore_index).
            intent: (B, n_intents) per-utterance intent logits, computed from
                each sequence's own CLS hidden state.
        """
        B, L = idx.shape
        assert L <= self.pos_emb_size, f"Sequenza troppo lunga: {L} > {self.pos_emb_size}"

        positions = torch.arange(L, device=idx.device)
        x = self.emb_dropout(self.token_embed(idx) + self.pos_embed(positions))

        causal_mask = self.mask[:L, :L].unsqueeze(0).unsqueeze(0)  # (1,1,L,L) broadcast over (B, n_heads)
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)

        x = self.out_dropout(x)
        slots = self.slot_out(x)  # (B, L, slots_size); CLS row is ignored by the slot loss (ignore_index=PAD_TOKEN)

        # Gather the CLS hidden state per example at its OWN real length minus
        # one (seq_lens[i] - 1), not at the fixed last absolute position
        # x[:, -1]. Sequences in the batch are padded to different effective
        # lengths than the batch's max length, so x[:, -1] would often point
        # at a PAD position instead of the actual trailing CLS token; indexing
        # by the per-example seq_lens is what correctly retrieves CLS for
        # every example regardless of how much padding follows it.
        cls_tokens = torch.stack([x[i, seq_lens[i] - 1] for i in range(B)])  # (B, d_model)
        intent = self.intent_out(cls_tokens)  # (B, n_intents)

        return slots, intent


def init_weights(mat):
    """Uniform initialization of Linear layers (small range, as used in the
    course lab), with biases initialized to a small constant rather than
    zero. Applied via model.apply(init_weights)."""
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def count_parameters(model):
    """Return the number of trainable (requires_grad=True) parameters,
    used to report model size (e.g. the 231K-parameter winning config)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
