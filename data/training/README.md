# Cross-encoder training data

## Automatic (weak labels)

`scripts/train_cross_encoder.py` builds pairs from:

- DDInter interaction chunks
- openFDA label sections (`drug_interactions`, `adverse_reactions`, etc.)
- Random negatives from unrelated chunks

## Manual (recommended for demo quality)

Copy `cross_encoder_pairs.jsonl.example` → `cross_encoder_pairs.jsonl` and add lines:

```json
{"query": "...", "passage": "...", "label": 1.0}
```

- `label`: `1.0` = relevant passage, `0.0` = not relevant
- `query`: same style as runtime (interaction / adverse / dosage questions)

## Train

```bash
export PYTHONPATH=src
python scripts/train_cross_encoder.py --epochs 2
```

Model saves to `models/cross_encoder_medlabel/`.

Set in `.env`:

```
CROSS_ENCODER_MODEL=models/cross_encoder_medlabel
```
