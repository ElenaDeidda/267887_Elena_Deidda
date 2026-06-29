# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
# model.py
# -----------------------------------------------------------------------------
# Part 1.A - GPT-2-style decoder-only Transformer language model, implemented
# FROM SCRATCH (no HuggingFace model classes) and trained on Penn TreeBank.
#
# Architecture follows the original GPT-2 design choices:
#   - Pre-norm Transformer blocks (LayerNorm applied BEFORE attention/feed-
#     forward, not after, unlike the original "Attention Is All You Need"
#     post-norm Transformer). Pre-norm keeps the residual stream un-normalized
#     end to end, which empirically gives much more stable gradients/training
#     for deep Transformers, especially when trained from scratch without
#     extensive warmup schedules or careful initialization.
#   - Causal self-attention via a triangular mask, pre-computed once and
#     registered as a buffer (see GPT2.__init__) rather than rebuilt on every
#     forward call, since the mask is constant given pos_emb_size.
#
# On top of the lab baseline, this file optionally supports (toggle via
# constructor arguments, used for the incremental hyperparameter search in
# Part 1.A):
#   - dropout at 4 distinct points (embedding sum, attention weights, the
#     attention output projection, and the feed-forward output)
#   - weight tying between token_embed and lm_head (input/output embedding
#     sharing, a la Press & Wolf 2017)
#
# Setting dropout=0.0 and weight_tying=False reproduces exactly the lab
# baseline, which made it easy to run the incremental experiments one change
# at a time and attribute any PPL change to a single factor.
#
# NOTE ON WEIGHT TYING RESULT: empirically, weight tying HURT performance on
# this setup (test PPL 33.07 without tying after 7 epochs vs 39.45 with tying
# requiring 20 epochs to converge). A plausible explanation is that forcing a
# single matrix to serve both as the input embedding lookup table and as the
# output (vocabulary-projection) layer creates conflicting gradient signals:
# the embedding wants to cluster semantically/distributionally similar tokens
# together, while the output projection wants to maximize separability for
# classification. This conflict appears to be worse here than in the original
# LSTM setting where weight tying was proposed, possibly because (a) this is a
# pre-norm Transformer trained from scratch (no pretraining to anchor the
# embedding geometry) and (b) PTB is a comparatively small corpus, so the
# shared matrix has fewer gradient updates to reconcile the two roles.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Masked (causal) multi-head self-attention block.

    Splits the d_model-dimensional projections of Q/K/V into n_heads
    independent heads of size h_dim = d_model // n_heads, computes scaled
    dot-product attention per head under a causal mask, then concatenates
    the heads back together and applies an output projection.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        # Linear projections for Query, Key, Value (one combined d_model -> d_model
        # projection each, later reshaped into per-head slices).
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final projection applied after concatenating all attention heads.
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout variant (Part 1.A): one dropout after the softmax attention
        # weights (regularizes which positions are attended to) and one after
        # the output projection (regularizes the residual contribution).
        # With dropout=0.0 both are no-ops, reproducing the baseline.
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Apply causal self-attention to a batch of token representations.

        Args:
            x: tensor (B, L, d_model), the input representations for this block.
            mask: causal mask slice (1, 1, L, L) (or broadcastable), 1 where
                attention is allowed (j <= i) and 0 where it must be blocked.

        Returns:
            Tensor (B, L, d_model): attention output, NOT yet added to the
            residual stream (the residual connection is added by the caller,
            TransformerBlock.forward).
        """
        B, L, d_model = x.size()  # batch, sequence length, model dimension

        q = self.w_q(x)  # (B, L, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape (B, L, d_model) -> (B, n_heads, L, h_dim) so each head can
        # attend independently.
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # Scaled dot-product similarity Q*K^T per head -> (B, n_heads, L, L)
        similarity = q @ k.transpose(-2, -1)

        # Scale by 1/sqrt(h_dim) (standard scaled dot-product attention) to
        # keep the softmax input variance roughly constant regardless of head
        # dimension, avoiding overly peaked/saturated softmax outputs.
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))

        # Causal mask: future positions (j > i) are set to -inf so that after
        # softmax they receive zero attention weight. This enforces the
        # autoregressive constraint required for language modeling.
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)
        # Dropout variant: randomly zero out some attention weights.
        attn = self.attn_dropout(attn)

        y = attn @ v                       # (B, n_heads, L, h_dim)
        y = y.transpose(1, 2)              # (B, L, n_heads, h_dim)
        y = y.contiguous().view(B, L, d_model)  # merge heads back together
        y = self.out_proj(y)
        # Dropout variant: applied to the attention block's output before it
        # is added back into the residual stream by the caller.
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network applied independently to each token.

    Standard GPT-2 MLP: Linear(d_model -> hidden_dim) -> GELU -> Linear(hidden_dim
    -> d_model). GELU (rather than ReLU) is used to match the GPT-2 design.
    """

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            # Dropout variant (Part 1.A): applied after the second linear
            # layer, i.e. on the feed-forward block's output before it is
            # added to the residual stream by TransformerBlock.forward.
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One GPT-2-style Transformer block using PRE-norm residual connections.

    Order is LayerNorm -> Attention -> add residual, then LayerNorm -> FeedForward
    -> add residual. This is "pre-norm" because LayerNorm is applied to the
    input of each sub-layer BEFORE attention/feed-forward, rather than to the
    sub-layer's output as in the original ("post-norm") Transformer. Pre-norm
    keeps an unimpeded residual path from input to output across all layers,
    which in practice yields much more stable training (better-behaved
    gradients, less sensitivity to learning rate/initialization) for deep
    stacks trained from scratch - important here since this model has no
    pretraining or learning-rate warmup schedule to compensate.
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        # Pre-norm: normalize BEFORE the sub-layer, then add its output back
        # onto the un-normalized residual stream `x`.
        x = x + self.attn(self.ln1(x), mask)  # attention + residual connection
        x = x + self.ff(self.ln2(x))          # feed-forward + residual connection
        return x


class GPT2(nn.Module):
    """From-scratch GPT-2-style decoder-only Transformer for language modeling.

    Embeds tokens and absolute positions, sums them, runs the result through
    a stack of pre-norm Transformer blocks under a causal mask, applies a
    final LayerNorm, and projects to vocabulary logits.

    Part 1.A experimental toggles:
        dropout:       dropout probability applied at 4 points in the network
                        (embedding sum, attention weights, attention output
                        projection, feed-forward output). 0.0 reproduces the
                        no-dropout lab baseline.
        weight_tying:  if True, lm_head.weight is set to alias token_embed.weight
                        (input/output embedding sharing). Empirically this HURT
                        results in this from-scratch pre-norm setup (see module
                        docstring for the full explanation) so the winning
                        config keeps this False.
    """

    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.0,
        weight_tying=False,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        # Learned token and (absolute) position embeddings, summed at the
        # input as in the original GPT-2.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # Dropout variant (Part 1.A): applied right after summing token and
        # position embeddings, before entering the Transformer stack.
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        # Output head: projects the final hidden states onto vocabulary logits.
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight tying variant (Part 1.A): share the same weight matrix
        # between the input token embedding and the output projection. This
        # is the classic Press & Wolf (2017) trick, originally shown to help
        # LSTM language models; here (pre-norm Transformer, trained from
        # scratch on PTB) it empirically hurt PPL and slowed convergence (see
        # module docstring), so the final winning config sets this to False.
        if weight_tying:
            self.lm_head.weight = self.token_embed.weight

        # Triangular causal mask: token i may only attend to tokens j with
        # j <= i. Built once for the maximum sequence length (pos_emb_size)
        # and registered as a buffer (not a learnable parameter) so it is
        # automatically moved to the right device by .to(device) and is not
        # recomputed on every forward call - it only needs to be sliced down
        # to the current sequence length L.
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        # Buffer (not a parameter): no gradient, but still follows .to(device).
        self.register_buffer("mask", mask)

    def forward(self, idx):
        """Compute next-token logits for a batch of input token ids.

        Args:
            idx: LongTensor (B, L) of input token ids (already shifted by the
                caller so that idx[:, t] is the model's input at position t;
                the corresponding target is provided separately by the
                training/eval loop in functions.py).

        Returns:
            Tensor (B, L, vocab_size) of unnormalized logits over the
            vocabulary for each position.
        """
        B, L = idx.shape
        assert L <= self.pos_emb_size, "Sequenza piu' lunga delle posizioni apprese"

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.emb_dropout(x)

        # Slice the precomputed causal mask down to the current sequence
        # length rather than rebuilding it.
        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits  # (B, L, vocab_size)
