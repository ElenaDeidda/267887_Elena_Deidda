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

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTforNLU(nn.Module):
    """BERT fine-tuned per intent + slot. Intent dal [CLS] (posizione 0, che BERT
    pre-addestra per aggregare l'intera sequenza); slot da ogni token."""

    def __init__(self, slots_size, n_intents,
                 model_name="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 base / 1024 large

        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Returns slots (B, L, slots_size), intent (B, n_intents)."""
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        last_hidden = output.last_hidden_state  # (B, L, hidden_size)

        cls_repr = last_hidden[:, 0, :]  # [CLS] in posizione 0

        slots = self.slot_out(self.dropout(last_hidden))  # (B, L, slots_size)
        intent = self.intent_out(self.dropout(cls_repr))  # (B, n_intents)
        return slots, intent


class GPT2forNLU(nn.Module):
    """GPT-2 fine-tuned per intent + slot. GPT-2 e' causale: l'intent si estrae dal
    token EOS appeso in CODA (l'unica posizione che ha visto tutta la frase); slot
    da ogni token."""

    def __init__(self, slots_size, n_intents,
                 model_name="openai-community/gpt2", dropout=0.1):
        super().__init__()
        self.gpt2 = AutoModel.from_pretrained(model_name)
        hidden_size = self.gpt2.config.n_embd  # 768 gpt2 / 1024 gpt2-medium

        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, slots_size)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, seq_lens):
        """seq_lens: (B,) lunghezza reale (incluso EOS). Returns slots, intent."""
        output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state  # (B, L, hidden_size)

        # vettore EOS/CLS alla posizione seq_lens[i]-1 (seq_lens conta da 1)
        cls_repr = torch.stack([
            last_hidden[i, seq_lens[i] - 1] for i in range(last_hidden.size(0))
        ])  # (B, hidden_size)

        slots = self.slot_out(self.dropout(last_hidden))  # (B, L, slots_size)
        intent = self.intent_out(self.dropout(cls_repr))  # (B, n_intents)
        return slots, intent


def count_parameters(model):
    """Numero di parametri addestrabili (utile per confrontare le varianti)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
