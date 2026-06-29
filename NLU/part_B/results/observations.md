# Osservazioni esperimenti - NLU Part 2.B (fine-tuning BERT/GPT-2)

_Generato automaticamente da main.py. Ultimo aggiornamento: 2026-06-29T05:23:16._

Metriche: **Slot F1** (conll) e **Intent Accuracy** sul test set (media +- std). Selezione degli iperparametri sulla **dev** slot F1.

## Migliore configurazione finora

- selezione su **dev F1: 0.9838** -> **test Slot F1: 0.9552 +- 0.0015**, **Intent Acc: 0.9746 +- 0.0014**
- esperimento: `bert_lr5e-05` (modello `bert-base-uncased`, lr=5e-05, dropout=0.1)

## bert

| esperimento | model | lr | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|
| bert_lr0.0001 | bert-base-uncased | 0.0001 | 0.1 | 0.9811 | 0.9559+-0.0014 | 0.9742+-0.0027 |  |
| bert_lr5e-05 | bert-base-uncased | 5e-05 | 0.1 | 0.9838 | 0.9552+-0.0015 | 0.9746+-0.0014 | * |
| bert_lr2e-05 | bert-base-uncased | 2e-05 | 0.1 | 0.9822 | 0.9544+-0.0023 | 0.9750+-0.0032 |  |
| bert_dropout0.1 | bert-base-uncased | 5e-05 | 0.1 | 0.9838 | 0.9552+-0.0015 | 0.9746+-0.0014 |  |
| bert_dropout0.3 | bert-base-uncased | 5e-05 | 0.3 | 0.9824 | 0.9519+-0.0037 | 0.9731+-0.0027 |  |
| bert-large_lr5e-05_dropout0.1 | bert-large-uncased | 5e-05 | 0.1 | 0.9826 | 0.9548+-0.0005 | 0.9754+-0.0016 |  |

- Osservazione (scelta sul dev): migliore `bert_lr5e-05` (dev F1 0.9838, test Slot F1 0.9552, Intent Acc 0.9746).
- Note (da completare nel report): 

## gpt2

| esperimento | model | lr | dropout | dev F1 | test Slot F1 | test Intent Acc | |
|---|---|---|---|---|---|---|---|
| gpt2_lr0.0001 | openai-community/gpt2 | 0.0001 | 0.1 | 0.9629 | 0.9142+-0.0053 | 0.9694+-0.0011 |  |
| gpt2_lr5e-05 | openai-community/gpt2 | 5e-05 | 0.1 | 0.9628 | 0.9145+-0.0036 | 0.9709+-0.0016 |  |
| gpt2_lr2e-05 | openai-community/gpt2 | 2e-05 | 0.1 | 0.9556 | 0.9055+-0.0090 | 0.9716+-0.0019 |  |
| gpt2_dropout0.1 | openai-community/gpt2 | 0.0001 | 0.1 | 0.9629 | 0.9142+-0.0053 | 0.9694+-0.0011 |  |
| gpt2_dropout0.3 | openai-community/gpt2 | 0.0001 | 0.3 | 0.9599 | 0.9158+-0.0053 | 0.9664+-0.0024 |  |
| gpt2-medium_lr0.0001_dropout0.1 | openai-community/gpt2-medium | 0.0001 | 0.1 | 0.9644 | 0.9125+-0.0016 | 0.9694+-0.0029 | * |

- Osservazione (scelta sul dev): migliore `gpt2-medium_lr0.0001_dropout0.1` (dev F1 0.9644, test Slot F1 0.9125, Intent Acc 0.9694).
- Note (da completare nel report): 

