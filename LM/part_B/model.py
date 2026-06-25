# model.py
# Part 1.B - GPT-2 pre-addestrato con adattatori LoRA (Low-Rank Adaptation) fatti a mano.
#
# Idea di LoRA (Hu et al., 2022): si congelano i pesi pre-addestrati W e si aggiunge
# un aggiornamento a basso rango  W(x) + (alpha/r) * B(A(x)),  con A: d->r e B: r->d.
# Init: A ~ N(0,1) e B = 0, cosi' all'inizio il delta B*A = 0 e il modello parte
# identico al GPT-2 pre-addestrato; durante il training si addestrano SOLO A e B.
#
# La consegna chiede di applicare LoRA alle matrici di query, key e value e di
# implementarlo a mano (niente librerie tipo PEFT).
#
#   CustomGPT2Attention : GPT2Attention + adattatori LoRA su Q/K/V
#   GPT2_LoRA           : GPT2LMHeadModel con i blocchi di attenzione sostituiti
#
# NB: il forward di CustomGPT2Attention replica quello di transformers==4.38.0,
# quindi va tenuta quella versione (vedi requirements.txt).

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


# ----------------------------------------------------------------------------
# 1. Attenzione GPT-2 con adattatori LoRA su Q, K, V
# ----------------------------------------------------------------------------

class CustomGPT2Attention(GPT2Attention):
    """GPT2Attention con LoRA su query/key/value.

    Eredita c_attn, c_proj, dropout e maschera da GPT2Attention (resteranno
    congelati) e aggiunge sei nn.Linear senza bias: per ognuna di Q/K/V una
    down-projection A (d->r) e una up-projection B (r->d). Lo scaling e' alpha/rank.
    """

    def __init__(self, config, rank, alpha):
        super().__init__(config)  # inizializza c_attn, c_proj, attn/resid dropout, bias causale

        embed_dim = config.hidden_size  # 768 per GPT-2 base

        self.rank = rank
        # scaling = alpha/rank: regola l'intensita' del delta in modo indipendente dal lr
        self.scaling = alpha / rank

        # down (d->r) e up (r->d) per Q, K, V
        self.lora_A_q = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, embed_dim, bias=False)
        self.lora_A_k = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_k = nn.Linear(rank, embed_dim, bias=False)
        self.lora_A_v = nn.Linear(embed_dim, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, embed_dim, bias=False)

        # init dal paper (sez. 4.1): A ~ N(0,1) (il gradiente scorre da subito),
        # B = 0  ->  delta = B*A = 0  ->  modello iniziale = pre-addestrato.
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
        """Forward identico a GPT2Attention di transformers==4.38.0, con in piu' i
        delta LoRA sommati a query/key/value nel ramo self-attention."""
        if encoder_hidden_states is not None:
            # ramo cross-attention: non usato nel decoder-only, lasciato invariato
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # self-attention: c_attn (congelata) proietta in 3*embed_dim, poi split Q/K/V
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            # delta LoRA addestrabile sommato alle proiezioni congelate
            query = query + self.lora_B_q(self.lora_A_q(hidden_states)) * self.scaling
            key   = key   + self.lora_B_k(self.lora_A_k(hidden_states)) * self.scaling
            value = value + self.lora_B_v(self.lora_A_v(hidden_states)) * self.scaling

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:  # KV-cache (generazione)
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
        attn_output = self.c_proj(attn_output)        # proiezione di output (congelata)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # (attn_output, present, [attn_weights])


# ----------------------------------------------------------------------------
# 2. GPT-2 con LoRA - modello completo
# ----------------------------------------------------------------------------

class GPT2_LoRA(GPT2LMHeadModel):
    """GPT-2 con adattatori LoRA iniettati in ogni blocco di attenzione.

    from_pretrained chiama prima __init__ (qui sostituiamo block.attn con
    CustomGPT2Attention) e poi carica i pesi pre-addestrati sull'intero modello,
    sovrascrivendo c_attn/c_proj/...; le matrici LoRA non sono nel checkpoint e
    mantengono l'inizializzazione (A~N(0,1), B=0). Il freeze (in functions.py)
    lascia addestrabili solo gli adapter.
    """

    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        # rank/alpha sono keyword-only: non vengono passati a GPT2LMHeadModel
        super().__init__(*model_args, **model_kwargs)

        # self.transformer.h = lista dei GPT2Block (12 per GPT-2 base)
        for block in self.transformer.h:
            old_attn = block.attn
            new_attn = CustomGPT2Attention(self.config, rank=rank, alpha=alpha)
            # strict=False: le matrici LoRA non sono nello state_dict di old_attn
            # -> mantengono l'init; c_attn/c_proj verranno da from_pretrained
            new_attn.load_state_dict(old_attn.state_dict(), strict=False)
            block.attn = new_attn

    def forward(self, *args, **kwargs):
        """Passthrough a GPT2LMHeadModel (gestisce logits, shift dei label e loss)."""
        return super().forward(*args, **kwargs)


# ----------------------------------------------------------------------------
# 3. Utilita'
# ----------------------------------------------------------------------------

def param_stats(model):
    """Stampa e restituisce (totali, addestrabili) dei parametri.

    Dopo il freeze solo le matrici LoRA sono addestrabili: per GPT-2 base sono
    circa lo 0.2-0.7% del totale.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parametri totali:     {total:,}")
    print(f"  parametri addestrabili: {trainable:,}")
    print(f"  parametri congelati:  {total - trainable:,}")
    print(f"  quota addestrabile:   {100 * trainable / total:.4f}%")
    return total, trainable
