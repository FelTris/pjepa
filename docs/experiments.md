# Training and evaluation

Run these commands from the repository root after installing P-JEPA and downloading
the [feature archives](data.md) and [encoder weights](../README.md#models).

LEMON training and the raw/student surgical linear probes:

```bash
python -m pjepa.cli.lemon_ssl --config configs/lemon/block_causal.json

python -m pjepa.cli.surgical_phase_linear_probe \
  --config configs/lemon/block_causal.json --dataset cholec80 \
  --protocol table1 --feature-source raw --results outputs/cholec80_raw.json

python -m pjepa.cli.surgical_phase_linear_probe \
  --config configs/lemon/block_causal.json --dataset cholec80 \
  --protocol table1 --feature-source student \
  --checkpoint "$PJEPA_CHECKPOINT_ROOT/lemon_plstitch_block32.pt" \
  --results outputs/cholec80_student.json
```

Use `--dataset m2cai16 --protocol official` for the complete M2CAI16 split.
These surgical commands **fit** frozen-feature heads. They do not update the
P-JEPA encoder. Surgical evaluation uses linear heads; LTContext probes are
excluded from the LEMON experiments. Saved linear-head cross-dataset transfer:

```bash
python -m pjepa.cli.surgical_phase_cross_dataset_eval \
  --config configs/lemon/cross_dataset_transfer.json
```

Assembly101 and EgoProceL use a common feature-experiment runner with explicit
actions. Training, fitting a probe, and scoring a saved head are different steps:

```bash
python -m pjepa.cli.feature_experiment train \
  --config configs/assembly101/block_causal_train.json
python -m pjepa.cli.feature_experiment fit-probe \
  --config configs/assembly101/block_causal_train.json \
  --checkpoint "$PJEPA_CHECKPOINT_ROOT/assembly101_tsm_block64.pt"
python -m pjepa.cli.feature_experiment evaluate \
  --config configs/assembly101/block_causal_train.json \
  --checkpoint "$PJEPA_CHECKPOINT_ROOT/assembly101_tsm_block64.pt" \
  --linear-head "$PJEPA_OUTPUT_ROOT/block_causal_train/linear.pt"
```

To fit the historical oracle temporal model, use `fit-probe` with
`configs/assembly101/oracle_clip_causal_ltcausal.json`. Evaluate the resulting
`linear.pt` and `temporal.pt` files with the same recipe. EgoProceL's
`oracle_fact_features_*` and `oracle_pooled_*` recipes use the same runner.
The corresponding `*_raw.json` recipes evaluate
unchanged input features. Feature SSL writes `best.pt` when a selection probe
improves and `last.pt` at every epoch. Loading an existing encoder for training
is a **warm start**; exact mid-run resume is not exposed as a supported command.

## Ablations

The release retains LEMON bidirectional training, full-video context budgets,
and temporal-order ablations. Context/order evaluation reuses a fixed
full-context linear head and re-encodes the input features. No figure-generation
scripts or historical result files are included.

```bash
python -m pjepa.cli.surgical_phase_context_sweep --help
python -m pjepa.cli.surgical_phase_order_ablation --help
```

The existing bidirectional training recipe uses 16 heads; the released
block-causal checkpoint was trained with 8. It is a historical recipe, not a
claim of a perfectly matched attention-only control. Any new matched control
must be trained and identified separately.


See [protocol notes](protocols.md) for split definitions, checkpoint selection,
and the assumptions of historical oracle models. Saved registered heads can be
loaded with `pjepa.probes.loading.load_probe`. Unregistered bare encoder state
dictionaries require explicit architecture metadata.
