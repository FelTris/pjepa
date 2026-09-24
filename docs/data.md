# Feature inputs and portable paths

The release starts from **precomputed features**, not videos. Supply the original
feature backbones, timing, and splits for each checkpoint. Archive feature
values are passed to the encoder without centering or standardization.
An arbitrary tensor with the correct width is sufficient for a smoke test, but
not a substitute for the original feature representation in an experiment.

The provided recipes expect this layout below `PJEPA_DATA_ROOT` (edit the recipe
paths to use another layout):

```text
lemon/pl_stitch_vitb16_4fps/sharded/index.pt
lemon/pl_stitch_vitb16_4fps/sharded/<shard files>
Cholec80/cholec80_pl_stitch_vitb16_1fps.pt
m2cai/m2cai16_pl_stitch_vitb16_1fps.pt
assembly101_small/assembly101_tsm_stride8.pt
assembly101_annotations/assembly101_coarse_segments.csv
egoprocel_small/fact_npy_features_4fps_full.pt
egoprocel_small/pooled_features_fact.pt
egoprocel_annotations/egoprocel_fact_features_segments.csv
egoprocel_annotations/egoprocel_segments.csv
```

Download the matching archives and segment manifests from
[FelTris/pjepa_features](https://huggingface.co/datasets/FelTris/pjepa_features):

```bash
python -m pip install huggingface_hub
export PJEPA_DATA_ROOT=/path/to/pjepa-data
hf download FelTris/pjepa_features --repo-type dataset --local-dir "$PJEPA_DATA_ROOT"
```

The dataset download supplies every path above, including the three CSV segment
manifests. The code repository contains no dataset tensors or annotations.
For extractor provenance, video coverage, timestamps, checksums, and selective
downloads, see the [dataset card](huggingface_dataset_card.md). Archive builders
remain available for converting your own feature files; they do not download data.

## Formats

- LEMON uses an index with `shards`, `shard_ids`, `local_indices`, `video_ids`,
  `paths`, `splits`, `num_tokens`, `feature_dim`, and `target_fps`. Shards live
  beside the index. Index-relative paths take precedence over an old stored
  absolute shard root. The training loader subsamples 4-fps features to 1 fps.
- Surgical phase archives contain `dataset`, `video_ids`, `source_splits`,
  `tokens`, `offsets`, `frame_indices`, `phase_labels`, `phase_class_names`,
  `feature_dim`, and `target_fps`. Labels and timestamps must align with tokens.
  Each downstream item is a complete video.
- Assembly101/EgoProceL packed archives contain `paths`, `tokens` (`total_tokens`
  by feature dimension), `times`, and `offsets`. Segment manifests supply
  dataset splits, labels, sequence IDs, and start/end times. The existing
  sample/take-key conventions and background handling are preserved.
- Direct inference consumes a torch dictionary with `features` (`B,T,D`), an
  optional Boolean `valid_mask` (`B,T`), and oracle `segment_lengths` (`B,S`)
  where applicable. Segment lengths sum to the number of valid tokens. Padding
  belongs at the end of each sequence.

## Builders

Use the builders as modules; each command documents its required inputs:

```bash
python -m pjepa.builders.surgical.pack_pl_stitch_archives --help
python -m pjepa.builders.surgical.pack_surgical_phase_archive --help
python -m pjepa.builders.build_assembly101_lmdb_archive --help
python -m pjepa.builders.build_assembly101_frame_archive --help
python -m pjepa.builders.build_egoprocel_fact_feature_manifest --help
python -m pjepa.builders.build_egoprocel_fact_npy_archive --help
```

The frame-archive builder also packs compatible per-video pooled feature files.
Pooled EgoProceL experiments still require their matching segment manifest;
FACT-feature manifests are built from their feature, ground-truth, and split
files. These builders do not import FACT, PL-Stitch, or V-JEPA video encoders.

Only load trusted torch archives/checkpoints. Some legacy dataset archives
contain metadata beyond tensors; the loaders preserve their format.
