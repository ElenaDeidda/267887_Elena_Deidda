# Osservazioni esperimenti - NLU Part 2.A (intent + slot)

_Generato automaticamente da main.py. Ultimo aggiornamento: 2026-06-28T19:42:46._

Metriche: **Slot F1** (conll) e **Intent Accuracy** sul test set (media +- std su piu' run). Selezione degli iperparametri sulla **dev** slot F1.

## Migliore configurazione finora

- selezione su **dev F1: 0.9414** -> **test Slot F1: 0.8876 +- 0.0057**, **Intent Acc: 0.9337 +- 0.0061**
- esperimento: `step1_nheads4`
- config: lr=0.01, d_model=64, n_heads=4, num_layers=2, ff_dim=256, dropout=0.0

## lr

| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|---|---|---|
| step0_lr0.1 | 0.1 | 64 | 2 | 2 | 256 | 0.0 | 0.2365 | 0.1838+-0.0255 | 0.7514+-0.0134 |  |
| step0_lr0.01 | 0.01 | 64 | 2 | 2 | 256 | 0.0 | 0.9286 | 0.8759+-0.0070 | 0.9212+-0.0169 | * |
| step0_lr0.001 | 0.001 | 64 | 2 | 2 | 256 | 0.0 | 0.9082 | 0.8535+-0.0174 | 0.8990+-0.0067 |  |
| step0_lr0.0001 | 0.0001 | 64 | 2 | 2 | 256 | 0.0 | 0.1209 | 0.0834+-0.0439 | 0.7077+-0.0000 |  |

- Osservazione (scelta sul dev): migliore `step0_lr0.01` (dev F1 0.9286, test Slot F1 0.8759).
- Note (da completare nel report): 

## d_model

| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|---|---|---|
| step1_dmodel128 | 0.01 | 128 | 2 | 2 | 256 | 0.0 | 0.8434 | 0.7928+-0.0370 | 0.8652+-0.0325 | * |
| step1_dmodel256 | 0.01 | 256 | 2 | 2 | 256 | 0.0 | 0.5121 | 0.4715+-0.0907 | 0.7935+-0.0158 |  |

- Osservazione (scelta sul dev): migliore `step1_dmodel128` (dev F1 0.8434, test Slot F1 0.7928).
- Note (da completare nel report): 

## n_heads

| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|---|---|---|
| step1_nheads4 | 0.01 | 64 | 4 | 2 | 256 | 0.0 | 0.9414 | 0.8876+-0.0057 | 0.9337+-0.0061 | * |
| step1_nheads8 | 0.01 | 64 | 8 | 2 | 256 | 0.0 | 0.9364 | 0.8874+-0.0055 | 0.9301+-0.0108 |  |

- Osservazione (scelta sul dev): migliore `step1_nheads4` (dev F1 0.9414, test Slot F1 0.8876).
- Note (da completare nel report): 

## num_layers

| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|---|---|---|
| step1_layers4 | 0.01 | 64 | 4 | 4 | 256 | 0.0 | 0.9341 | 0.8858+-0.0083 | 0.9382+-0.0125 | * |
| step1_layers6 | 0.01 | 64 | 4 | 6 | 256 | 0.0 | 0.9312 | 0.8792+-0.0172 | 0.9299+-0.0086 |  |

- Osservazione (scelta sul dev): migliore `step1_layers4` (dev F1 0.9341, test Slot F1 0.8858).
- Note (da completare nel report): 

## ff_dim

| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|---|---|---|
| step1_ffdim512 | 0.01 | 64 | 4 | 2 | 512 | 0.0 | 0.9261 | 0.8675+-0.0154 | 0.9256+-0.0160 |  |
| step1_ffdim1024 | 0.01 | 64 | 4 | 2 | 1024 | 0.0 | 0.9322 | 0.8796+-0.0063 | 0.9306+-0.0073 | * |

- Osservazione (scelta sul dev): migliore `step1_ffdim1024` (dev F1 0.9322, test Slot F1 0.8796).
- Note (da completare nel report): 

## dropout

| esperimento | lr | d_model | n_heads | layers | ff_dim | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|---|---|---|
| step2_dropout0.1 | 0.01 | 64 | 4 | 2 | 256 | 0.1 | 0.9357 | 0.8984+-0.0085 | 0.9458+-0.0043 |  |
| step2_dropout0.2 | 0.01 | 64 | 4 | 2 | 256 | 0.2 | 0.9403 | 0.9073+-0.0034 | 0.9557+-0.0102 | * |
| step2_dropout0.3 | 0.01 | 64 | 4 | 2 | 256 | 0.3 | 0.9321 | 0.9028+-0.0058 | 0.9574+-0.0034 |  |

- Osservazione (scelta sul dev): migliore `step2_dropout0.2` (dev F1 0.9403, test Slot F1 0.9073).
- Note (da completare nel report): 

