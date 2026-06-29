# model.py
# Part 1.B - Pre-trained GPT-2 with a hand-rolled LoRA (Low-Rank Adaptation) implementation.
#
# ---------------------------------------------------------------------------------------
# MODULE OVERVIEW
# ---------------------------------------------------------------------------------------
# LoRA (Hu et al., 2022) freezes the pre-trained weight matrix W and learns a low-rank
# correction on top of it:
#
#       h = W(x) + (alpha / r) * B(A(x))      with A: d -> r  and  B: r -> d
#
# Only A and B (a tiny number of parameters compared to W) are trained; W itself never
# receives gradients. This file implements LoRA "by hand" (no PEFT/other adapter
# libraries are used, per the assignment requirements) and applies it specifically to
# the query, key and value projections of GPT-2's self-attention blocks.
#
# Initialization is the crux of why LoRA works as a safe drop-in: A is drawn from
# N(0, 1) (gradients are non-zero from step 1, matching the original LoRA paper's
# convention) while B is initialized to all zeros. Since the injected delta is the
# product B @ A, starting with B = 0 forces delta = 0 at step 0 -- the adapted model is
# therefore mathematically IDENTICAL to the unmodified pre-trained GPT-2 the moment
# training begins, and only gradually drifts away from it as B (and A) are updated.
#
# Classes defined here:
#   CustomGPT2Attention : drop-in replacement for HuggingFace's GPT2Attention that adds
#                         LoRA adapters on the Q/K/V projections.
#   GPT2_LoRA           : GPT2LMHeadModel subclass that swaps every attention block for
#                         a CustomGPT2Attention instance at construction time.
#
# NOTE: CustomGPT2Attention.forward() intentionally mirrors the internals of
# transformers==4.38.0's GPT2Attention.forward() line-for-line (so that the only
# behavioral difference is the added LoRA delta) -- pin that transformers version
# (see requirements.txt), since HuggingFace has changed this forward signature/body
# in later releases.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


# ----------------------------------------------------------------------------
# 1. GPT-2 attention with LoRA adapters on Q, K, V
# ----------------------------------------------------------------------------

class CustomGPT2Attention(GPT2Attention):
    """GPT2Attention augmented with LoRA adapters on the query/key/value projections.

    Inherits c_attn, c_proj, the dropout layers and the causal mask buffer from
    GPT2Attention unchanged -- these will be frozen during training (see
    functions.freeze_pretrained_and_enable_lora). On top of that it adds six bias-free
    nn.Linear layers: for each of Q, K, V a down-projection A (embed_dim -> rank) and an
    up-projection B (rank -> embed_dim). The two are combined into a single additive
    correction term, scaled by alpha/rank so that the magnitude of the adaptation can be
    tuned independently of the learning rate.
    """

    def __init__(self, config, rank, alpha):
        super().__init__(config)  # sets up c_attn, c_proj, attn/resid dropout, causal bias buffer

        embed_dim = config.hidden_size  # 768 for GPT-2 base

        self.rank = rank
        # scaling = alpha / rank: controls how strongly the LoRA delta perturbs the
        # frozen projection, independently of the optimizer's learning rate.
        self.scaling = alpha / rank

        # Down-projection (d -> r) and up-projection (r -> d) for each of Q, K, V.
        self.lora_A_q = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, embed_dim, bias=False)
        self.lora_A_k = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_k = nn.Linear(rank, embed_dim, bias=False)
        self.lora_A_v = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, embed_dim, bias=False)

        # Initialization scheme from the original LoRA paper (sec. 4.1):
        #   - A ~ N(0, 1): non-zero from the start, so gradients can flow through A
        #     immediately once training starts.
        #   - B = 0: forces the injected delta = B @ A to be exactly zero at init, which
        #     guarantees the adapted model is numerically identical to the unmodified
        #     pre-trained GPT-2 at the very first forward pass. Training then gradually
        #     moves B (and A) away from this no-op starting point.
        for lora_A in (self.lora_A_q, self.lora_A_k, self.lora_A_v):
            nn.init.normal_(lora_A.weight)
        for lora_B in (self.lora_B_q, self.lora_B_k, self.lora_B_v):
            nn.init.zeros_(lora_B.weight)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        """Forward pass identical to transformers==4.38.0's GPT2Attention.forward(),
        with the LoRA deltas additionally summed into query/key/value in the
        self-attention branch (the cross-attention branch, unused for this decoder-only
        LM, is left untouched and carries no LoRA adapters)."""
        if encoder_hidden_states is not None:
            # Cross-attention branch: not exercised by this decoder-only LM (GPT-2 has
            # no encoder), kept verbatim from the parent class for API compatibility.
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # Self-attention: the frozen c_attn linear layer projects hidden_states to
            # 3*embed_dim in one go, then the result is split into Q, K, V.
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            # Trainable LoRA delta added on top of the frozen projections: for each of
            # Q/K/V, hidden_states is pushed through the down-projection A, then the
            # up-projection B, and scaled by alpha/rank before being added in. At
            # init (B=0) this entire term is zero (see __init__ for why).
            query = query + self.lora_B_q(self.lora_A_q(hidden_states)) * self.scaling
            key   = key   + self.lora_B_k(self.lora_A_k(hidden_states)) * self.scaling
            value = value + self.lora_B_v(self.lora_A_v(hidden_states)) * self.scaling

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:  # KV-cache path (used during autoregressive generation)
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)        # output projection (frozen, no LoRA here)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # (attn_output, present, [attn_weights])


# ----------------------------------------------------------------------------
# 2. GPT-2 with LoRA - full model
# ----------------------------------------------------------------------------

class GPT2_LoRA(GPT2LMHeadModel):
    """GPT-2 language model with LoRA adapters injected into every attention block.

    Construction/loading order matters here: HuggingFace's `from_pretrained()` first
    calls `__init__` (which is where this class replaces each block's `attn` with a
    `CustomGPT2Attention` instance), and only afterwards loads the pre-trained
    checkpoint weights into the resulting module tree. That checkpoint load overwrites
    c_attn/c_proj/etc. with the pre-trained values, but it has no entries for the
    lora_* parameters (they don't exist in the original GPT-2 checkpoint), so they are
    left untouched at their __init__-time initialization (A ~ N(0,1), B = 0). Freezing
    the backbone and leaving only the LoRA adapters trainable is handled separately in
    functions.freeze_pretrained_and_enable_lora, not here.
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        # rank/alpha are keyword-only and consumed here; they are never forwarded to
        # GPT2LMHeadModel.__init__, which does not know about them.
        super().__init__(*model_args, **model_kwargs)

        # self.transformer.h is the list of GPT2Block modules (12 for GPT-2 base).
        for block in self.transformer.h:
            old_attn = block.attn
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            # strict=False because old_attn's state_dict has no lora_* keys: this just
            # copies over c_attn/c_proj/bias/etc. from the (randomly-initialized, at
            # this point) old attention module, while leaving the freshly-initialized
            # LoRA matrices alone. The *real* pre-trained backbone weights arrive
            # afterwards, when from_pretrained() loads the checkpoint into the whole
            # model -- this load_state_dict call only preserves continuity of the
            # non-LoRA buffers/parameters across the module swap.
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            block.attn = new_attn

    def forward(self, *args, **kwargs):
        """Passthrough to GPT2LMHeadModel.forward(). HuggingFace's implementation
        internally shifts `labels` against the logits by one position and computes the
        cross-entropy loss when `labels` is provided -- see functions.py, which relies
        on this behavior instead of shifting labels manually."""
        return super().forward(*args, **kwargs)


# ----------------------------------------------------------------------------
# 3. Utilities
# ----------------------------------------------------------------------------

def param_stats(model):
    """Print and return (total_params, trainable_params) for `model`.

    After freeze_pretrained_and_enable_lora has run, only the LoRA matrices are
    trainable; for GPT-2 base this is typically around 0.2-0.7% of all parameters,
    depending on the chosen rank.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parametri totali:     {total:,}")
    print(f"  parametri addestrabili: {trainable:,}")
    print(f"  parametri congelati:  {total - trainable:,}")
    print(f"  quota addestrabile:   {100 * trainable / total:.4f}%")
    return total, trainable
