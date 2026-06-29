# Osservazioni esperimenti - LM Part 1.B (LoRA)

_Generato automaticamente da main.py. Ultimo aggiornamento: 2026-06-29T11:44:06._

Vincolo della consegna: **PPL test < 250**.

## Migliore configurazione finora

- selezione su **best dev PPL: 21.03** -> **Test PPL: 19.14** (OK <250)
- esperimento: `step2_alpha_half`
- rank: `16` | alpha: `8` | scaling: `0.5` | lr: `0.0005`
- parametri addestrabili (LoRA): 884,736

## Step 0
_Step 0 - Ricerca del learning rate (rank/alpha fissi)._

| esperimento | rank | alpha | scaling | lr | epoche | dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|---|---|
| step0_lr1e-3 | 4 | 8 | 2.0 | 0.001 | 17 | 23.75 | 21.47 | si |  |
| step0_lr5e-4 | 4 | 8 | 2.0 | 0.0005 | 17 | 23.27 | 21.02 | si | * |
| step0_lr1e-4 | 4 | 8 | 2.0 | 0.0001 | 20 | 23.95 | 21.65 | si |  |

- Osservazione (scelta sul dev): migliore `step0_lr5e-4` (dev PPL 23.27, test PPL 21.02).
- Note (da completare nel report): 

## Step 1
_Step 1 - Rango r di LoRA (alpha = 2*r, scaling = 2.0)._

| esperimento | rank | alpha | scaling | lr | epoche | dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|---|---|
| step1_rank4 | 4 | 8 | 2.0 | 0.0005 | 17 | 23.27 | 21.02 | si |  |
| step1_rank8 | 8 | 16 | 2.0 | 0.0005 | 20 | 21.99 | 19.91 | si |  |
| step1_rank16 | 16 | 32 | 2.0 | 0.0005 | 19 | 21.20 | 19.35 | si | * |

- Osservazione (scelta sul dev): migliore `step1_rank16` (dev PPL 21.20, test PPL 19.35).
- Note (da completare nel report): 

## Step 2
_Step 2 - Alpha (a parita' del miglior rank; scaling = alpha/rank)._

| esperimento | rank | alpha | scaling | lr | epoche | dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|---|---|
| step2_alpha_half | 16 | 8 | 0.5 | 0.0005 | 20 | 21.03 | 19.14 | si | * |

- Osservazione (scelta sul dev): migliore `step2_alpha_half` (dev PPL 21.03, test PPL 19.14).
- Note (da completare nel report): 

