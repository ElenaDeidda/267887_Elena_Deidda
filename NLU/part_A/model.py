# model.py
# Part 2.A - GPT-2 from scratch per Intent Classification + Slot Filling (joint).
#
# Stesso backbone della Part 1.A (multi-head causal attention, feed-forward,
# blocchi Pre-LN, embedding di token + posizione, maschera causale) ma con due
# teste di output al posto della lm_head:
#   slot_out   : Linear(d_model, slots_size) - una etichetta per token (sequence labeling)
#   intent_out : Linear(d_model, n_intents)  - una etichetta per frase, dal token CLS
#
# Il CLS e' appeso in CODA alla frase: GPT-2 e' causale, quindi solo l'ultima
# posizione ha "visto" tutta la frase (un [CLS] iniziale, come in BERT, sarebbe
# poco informativo). Il target slot del CLS e' PAD, quindi e' ignorato dalla loss.
#
# Modifiche incrementali della consegna (2.A):
#   - ricerca di d_model, n_heads, num_layers, ff_dim (in main.py)
#   - dropout (controllato dal parametro `dropout`), incluso un dropout prima
#     delle teste di output.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Self-attention multi-testa causale (come nella Part 1.A)."""

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
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, L, d_k)

    def _merge_heads(self, x):
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, x, mask=None):
        Q = self._split_heads(self.w_q(x))
        K = self._split_heads(self.w_k(x))
        V = self._split_heads(self.w_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:        # maschera additiva: -inf sulle posizioni future
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V)                 # (B, n_heads, L, d_k)
        out = self.out_proj(self._merge_heads(context))
        return self.proj_dropout(out)


class FeedForward(nn.Module):
    """Feed-forward posizionale: Linear -> GELU -> Linear -> Dropout."""

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
    """Blocco Pre-LN: x = x + Attn(LN(x)); x = x + FF(LN(x))."""

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
    """GPT-2 from scratch con due teste (slot per token, intent dal CLS in coda)."""

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

        # embedding: token (include pad=0, unk=1, cls=2) + posizione appresa
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # MODIFICA (consegna 2.A): dropout prima delle teste di output
        self.out_dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(d_model, slots_size)     # una etichetta per token
        self.intent_out = nn.Linear(d_model, n_intents)    # una etichetta per frase (CLS)

        # maschera causale come buffer: la riga i puo' attendere solo j <= i.
        # Il CLS in coda vede tutta la frase -> adatto ad aggregare per l'intent.
        mask = torch.triu(torch.full((pos_emb_size, pos_emb_size), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        """idx: (B, L) token id (con padding e CLS); seq_lens: (B,) lunghezza reale (incl. CLS).
        Returns: slots (B, L, slots_size), intent (B, n_intents)."""
        B, L = idx.shape
        assert L <= self.pos_emb_size, f"Sequenza troppo lunga: {L} > {self.pos_emb_size}"

        positions = torch.arange(L, device=idx.device)
        x = self.emb_dropout(self.token_embed(idx) + self.pos_embed(positions))

        causal_mask = self.mask[:L, :L].unsqueeze(0).unsqueeze(0)  # (1,1,L,L) broadcast
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)

        x = self.out_dropout(x)
        slots = self.slot_out(x)  # (B, L, slots_size); il CLS verra' ignorato dalla loss

        # vettore CLS: ultimo token reale di ogni sequenza (indice seq_lens[i]-1),
        # non x[:, -1] perche' le sequenze sono paddizzate a lunghezze diverse
        cls_tokens = torch.stack([x[i, seq_lens[i] - 1] for i in range(B)])  # (B, d_model)
        intent = self.intent_out(cls_tokens)  # (B, n_intents)

        return slots, intent


def init_weights(mat):
    """Inizializzazione uniforme dei layer lineari (come nel laboratorio)."""
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def count_parameters(model):
    """Numero di parametri addestrabili."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
