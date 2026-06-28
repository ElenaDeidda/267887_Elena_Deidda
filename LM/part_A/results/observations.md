# Osservazioni esperimenti - LM Part 1.A

_Generato automaticamente da main.py. Ultimo aggiornamento: 2026-06-28T18:20:42._

Vincolo della consegna: **PPL test < 250**. La modifica va tenuta solo se migliora (o non peggiora) la PPL; gli esperimenti falliti vanno comunque commentati nel report.

## Migliore configurazione finora

- selezione su **best dev PPL: 36.86** -> **Test PPL: 33.07** (OK <250)
- gruppo: `weight_tying` | label: `weight_tying=False`
- lr: `0.0005` | parametri: 52,050,769
- config: `{'pos_emb_size': 1024, 'd_model': 384, 'n_heads': 8, 'num_layers': 6, 'ff_dim': 2048, 'dropout': 0.1, 'weight_tying': False}`

## baseline-lr
_Step 0 - Baseline: ricerca del learning rate (modello fisso)._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.001 | 29,203,537 | 6 | 39.43 | 35.63 | sì |  |
| 0.0005 | 0.0005 | 29,203,537 | 7 | 39.13 | 35.68 | sì | ⭐ |
| 0.0003 | 0.0003 | 29,203,537 | 8 | 39.25 | 35.57 | sì |  |
| 0.0001 | 0.0001 | 29,203,537 | 17 | 40.56 | 36.61 | sì |  |

- Osservazione: il valore migliore (scelto sul dev) e' **0.0005** (dev PPL 39.13, test PPL 35.68); il peggiore 0.0001 (dev PPL 40.56), un divario di 1.43 punti di dev PPL.
- Note (da completare nel report): 

## d_model
_Step 1 - Iperparametri: dimensione del modello (d_model)._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| 128 | 0.0005 | 14,366,801 | 10 | 41.55 | 37.42 | sì |  |
| 256 | 0.0005 | 29,203,537 | 7 | 39.13 | 35.68 | sì |  |
| 384 | 0.0005 | 44,564,561 | 6 | 39.12 | 35.32 | sì | ⭐ |

- Osservazione: il valore migliore (scelto sul dev) e' **384** (dev PPL 39.12, test PPL 35.32); il peggiore 128 (dev PPL 41.55), un divario di 2.43 punti di dev PPL.
- Note (da completare nel report): 

## n_heads
_Step 1 - Iperparametri: numero di teste di attenzione (n_heads)._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| 2 | 0.0005 | 44,564,561 | 6 | 39.47 | 35.47 | sì |  |
| 4 | 0.0005 | 44,564,561 | 6 | 39.12 | 35.32 | sì |  |
| 8 | 0.0005 | 44,564,561 | 6 | 38.71 | 35.03 | sì | ⭐ |

- Osservazione: il valore migliore (scelto sul dev) e' **8** (dev PPL 38.71, test PPL 35.03); il peggiore 2 (dev PPL 39.47), un divario di 0.76 punti di dev PPL.
- Note (da completare nel report): 

## num_layers
_Step 1 - Iperparametri: numero di blocchi transformer (num_layers)._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| 2 | 0.0005 | 41,803,089 | 6 | 41.56 | 37.57 | sì |  |
| 4 | 0.0005 | 44,564,561 | 6 | 38.71 | 35.03 | sì |  |
| 6 | 0.0005 | 47,326,033 | 6 | 37.99 | 34.66 | sì | ⭐ |

- Osservazione: il valore migliore (scelto sul dev) e' **6** (dev PPL 37.99, test PPL 34.66); il peggiore 2 (dev PPL 41.56), un divario di 3.57 punti di dev PPL.
- Note (da completare nel report): 

## ff_dim
_Step 1 - Iperparametri: dimensione del feed-forward (ff_dim)._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| 512 | 0.0005 | 44,963,665 | 6 | 38.86 | 35.28 | sì |  |
| 1024 | 0.0005 | 47,326,033 | 6 | 37.99 | 34.66 | sì |  |
| 2048 | 0.0005 | 52,050,769 | 6 | 37.39 | 33.84 | sì | ⭐ |
| 512 | 0.0005 | 44,963,665 | 6 | 38.86 | 35.28 | sì |  |
| 1024 | 0.0005 | 47,326,033 | 6 | 37.99 | 34.66 | sì |  |
| 2048 | 0.0005 | 52,050,769 | 6 | 37.39 | 33.84 | sì |  |

- Osservazione: il valore migliore (scelto sul dev) e' **2048** (dev PPL 37.39, test PPL 33.84); il peggiore 512 (dev PPL 38.86), un divario di 1.47 punti di dev PPL.
- Note (da completare nel report): 

## dropout
_Step 2 - Dropout nei 4 punti della rete._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0005 | 52,050,769 | 6 | 37.48 | 33.82 | sì |  |
| 0.1 | 0.0005 | 52,050,769 | 7 | 37.46 | 33.72 | sì | ⭐ |
| 0.2 | 0.0005 | 52,050,769 | 8 | 37.75 | 33.65 | sì |  |

- Osservazione: il valore migliore (scelto sul dev) e' **0.1** (dev PPL 37.46, test PPL 33.72); il peggiore 0.2 (dev PPL 37.75), un divario di 0.29 punti di dev PPL.
- Note (da completare nel report): 

## weight_tying
_Step 3 - Weight tying tra token_embed e lm_head._

| valore | lr | params | epoche | best dev PPL | test PPL | <250 | |
|---|---|---|---|---|---|---|---|
| False | 0.0005 | 52,050,769 | 7 | 36.86 | 33.07 | sì | ⭐ |
| True | 0.0005 | 32,752,081 | 20 | 44.47 | 39.45 | sì |  |

- Osservazione: il valore migliore (scelto sul dev) e' **False** (dev PPL 36.86, test PPL 33.07); il peggiore True (dev PPL 44.47), un divario di 7.61 punti di dev PPL.
- Note (da completare nel report): 

