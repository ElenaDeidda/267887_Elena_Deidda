# Osservazioni esperimenti - LM Part 1.A

_Generato automaticamente da main.py. Ultimo aggiornamento: 2026-06-27T21:15:38._

Vincolo della consegna: **PPL test < 250**. La modifica va tenuta solo se migliora (o non peggiora) la PPL; gli esperimenti falliti vanno comunque commentati nel report.

## Migliore configurazione finora

- selezione su **best dev PPL: 38.71** -> **Test PPL: 35.03** (OK <250)
- gruppo: `n_heads` | label: `n_heads=8`
- lr: `0.0005` | parametri: 44,564,561
- config: `{'pos_emb_size': 1024, 'd_model': 384, 'n_heads': 8, 'num_layers': 4, 'ff_dim': 1024, 'dropout': 0.0, 'weight_tying': False}`

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

