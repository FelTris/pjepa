# Feature archive terms and attribution

This collection combines derived features and annotations from several sources.
It is not offered under a blanket MIT dataset license. The MIT license for the
P-JEPA code and encoder weights does not replace source-data or feature-extractor
terms. Follow the terms of each source for the subset you use.

- LEMON: https://github.com/visurg-ai/LEMON . The upstream release licenses its
  metadata, video-ID lists, and annotations under CC BY 4.0; original video
  copyrights remain with their creators. This is not a claim that raw videos
  are licensed under CC BY 4.0.
- PL-Stitch: https://github.com/visurg-ai/PL-Stitch . Attribution for the
  pretrained representation used for LEMON, Cholec80, and M2CAI16.
- Cholec80 and M2CAI16: https://camma.unistra.fr/datasets/ . Original dataset
  access/use terms remain applicable to their annotations and derived data.
- Assembly101: https://github.com/assembly-101/assembly101-temporal-action-segmentation .
  The upstream dataset/TSM release specifies CC BY-NC 4.0. This collection
  repacks and subsamples the features and includes the matching segment manifest.
- EgoProceL features distributed with FACT:
  https://github.com/ZijiaLewisLu/CVPR2024-FACT . FACT's code license does not
  establish a blanket license for its bundled third-party datasets. EgoProceL
  includes source datasets identified in the CSV manifests; their terms apply.
- V-JEPA 2.1: https://github.com/facebookresearch/vjepa2 . Attribution for the
  frozen backbone used by the supervised EgoProceL framewise pooler.

The dataset card and manifest describe the archive transformations, provenance,
coverage, and known limitations. No raw videos are included.
