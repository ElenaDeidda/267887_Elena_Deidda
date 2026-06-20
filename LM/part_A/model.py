# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
# model.py
# Definizione dell'architettura GPT-2 (decoder-only) in PyTorch.
#
# Rispetto al baseline del laboratorio, qui sono integrate (in modo OPZIONALE,
# tramite parametri) le modifiche richieste dalla Part 1.A:
#   - dropout in 4 punti (embedding, pesi di attenzione, output proj, feed forward)
#   - weight tying tra token_embed e lm_head
#
# Tenendo dropout=0.0 e weight_tying=False si ottiene esattamente il baseline,
# utile per eseguire gli esperimenti incrementali uno alla volta.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Self-attention multi-testa mascherata (causale)."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        # proiezioni lineari per Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # proiezione finale dopo la concatenazione delle teste
        self.out_proj = nn.Linear(d_model, d_model)

        # MODIFICA (dropout): dopo i pesi di attenzione e dopo la output projection
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, d_model = x.size()  # batch, lunghezza sequenza, dimensione modello

        q = self.w_q(x)  # (B, L, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # reshape in (B, n_heads, L, h_dim)
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # similarita' Q*K^T per ogni testa -> (B, n_heads, L, L)
        similarity = q @ k.transpose(-2, -1)

        # normalizzazione per la radice della dimensione della testa
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))

        # maschera causale: le posizioni future vengono messe a -inf
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)
        # MODIFICA (dropout): dopo aver calcolato i pesi di attenzione
        attn = self.attn_dropout(attn)

        y = attn @ v                       # (B, n_heads, L, h_dim)
        y = y.transpose(1, 2)              # (B, L, n_heads, h_dim)
        y = y.contiguous().view(B, L, d_model)  # concatena le teste
        y = self.out_proj(y)
        # MODIFICA (dropout): dopo la output projection
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    """Rete feed-forward applicata indipendentemente a ogni token. Usa GELU."""

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            # MODIFICA (dropout): dopo l'ultimo layer lineare del Feed Forward
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Blocco transformer di GPT-2: LayerNorm -> Attn -> residuo, poi LayerNorm -> FF -> residuo."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)  # attenzione + connessione residuale
        x = x + self.ff(self.ln2(x))          # feed forward + connessione residuale
        return x


class GPT2(nn.Module):
    """Modello GPT-2 decoder-only costruito da zero.

    Parametri delle modifiche Part 1.A:
        dropout:       probabilita' di dropout (0.0 = baseline senza dropout)
        weight_tying:  se True, token_embed e lm_head condividono gli stessi pesi
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

        # embedding apprendibili di token e di posizione
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # MODIFICA (dropout): subito dopo la somma degli embedding
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        # layer di output: proietta sullo spazio del vocabolario
        self.lm_head = nn.Linear(d_model, vocab_size)

        # MODIFICA (weight tying): condivide gli stessi pesi tra input embedding e output
        if weight_tying:
            self.lm_head.weight = self.token_embed.weight

        # maschera triangolare: il token i puo' attendere solo i token j con j <= i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        # non e' un parametro apprendibile ma deve seguire il modello con .to(device)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        B, L = idx.shape
        assert L <= self.pos_emb_size, "Sequenza piu' lunga delle posizioni apprese"

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.emb_dropout(x)

        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits  # (B, L, vocab_size)
