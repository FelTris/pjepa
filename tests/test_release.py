import copy
import json

import pytest
import torch

from pjepa.checkpoints import load_model
from pjepa.inference.encoder import encode
from pjepa.models.feature_vjepa_rope import FeatureJEPA
from pjepa.training.feature_ssl import Trainer
from pjepa.evaluation.linear import evaluate_linear


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def architecture(**updates):
    result = dict(
        d_in=32,
        d_model=32,
        enc_depth=1,
        pred_depth=1,
        enc_heads=4,
        pred_heads=4,
        drop_path_prob=0.0,
        student_encoder_attention="block_causal",
        student_encoder_block_size=4,
        rope_mode="1d_flat",
    )
    return result | updates


def test_checkpoint_metadata_rejects_shape_compatible_mismatch(tmp_path):
    arch = architecture()
    model = FeatureJEPA(**arch)
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "architecture": arch}, path)
    restored, meta = load_model(path)
    with pytest.raises(ValueError, match="conflicts"):
        load_model(path, architecture=architecture(enc_heads=8, pred_heads=8))
    x = torch.randn(2, 11, 32)
    valid = torch.arange(11)[None, :] < torch.tensor([11, 7])[:, None]
    torch.testing.assert_close(
        encode(model.eval(), x, valid), encode(restored, x, valid), rtol=0, atol=0
    )
    assert meta["architecture"] == arch


def test_oracle_boundaries_required():
    model = FeatureJEPA(
        **architecture(student_encoder_attention="clip_causal", rope_mode="2d_clip_frame")
    )
    x = torch.randn(1, 11, 32)
    with pytest.raises(ValueError, match="oracle"):
        encode(model, x)
    y = encode(model, x, segment_lengths=torch.tensor([[3, 5, 3]]))
    assert y.shape == (1, 11, 32) and torch.isfinite(y).all()
    with pytest.raises(ValueError, match="sum"):
        encode(model, x, segment_lengths=torch.tensor([[3, 5, 4]]))


def test_encoder_preserves_inputs_and_applies_each_block_once():
    model = FeatureJEPA(**architecture(enc_depth=3)).eval()
    x = torch.randn(2, 1, 11, 32) * 7 + 13
    valid = torch.ones(2, 1, 11, dtype=torch.bool)
    for encoder in (model.student_enc, model.build_teacher()):
        calls = []
        projected_inputs = []
        handles = [
            encoder.proj.register_forward_pre_hook(
                lambda module, args: projected_inputs.append(args[0].detach().clone())
            )
        ]
        for index, block in enumerate(encoder.blocks):
            handles.append(
                block.register_forward_hook(
                    lambda module, args, output, index=index: calls.append(index)
                )
            )
        with torch.no_grad():
            encoder(x, valid_mask=valid, context_mask=valid, N=1, L=11)
        for handle in handles:
            handle.remove()
        assert calls == [0, 1, 2]
        torch.testing.assert_close(projected_inputs[0], x.flatten(1, 2), rtol=0, atol=0)


def test_ssl_step_has_finite_gradients_and_updates_teacher():
    torch.manual_seed(19)
    model = FeatureJEPA(**architecture())
    teacher = model.build_teacher()
    x = torch.randn(2, 1, 11, 32)
    valid = torch.arange(11)[None, None, :] < torch.tensor([11, 7])[:, None, None]
    target = valid.clone()
    target[:, :, ::2] = False
    before = copy.deepcopy(teacher.state_dict())
    out = model(
        {"x": x, "valid_mask": valid, "context_mask": valid & ~target, "target_mask": target},
        teacher=teacher,
    )
    loss = (out["preds_at_targets"] - out["teacher_at_targets"]).abs().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    from pjepa.models.ema import EMA

    EMA(momentum=0.9).update(teacher=teacher, student=model.student_enc)
    assert any(not torch.equal(before[k], v) for k, v in teacher.state_dict().items())


class TinyGenerator:
    """Two variable-length labeled takes using the packed-loader contract."""

    def __init__(self):
        g = torch.Generator().manual_seed(13)
        self.x = torch.randn(2, 12, 32, generator=g)
        self.y = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2, -100, -100, -100, -100]]
        )
        self.lengths = torch.tensor([[3, 3, 3, 3], [2, 3, 3, 0]])
        self.list_of_examples = ["a", "b"]
        self.reset()

    def reset(self):
        self.done = False

    def has_next(self):
        return not self.done

    def next_batch(self, batch_size):
        self.done = True
        return (
            self.x.clone(),
            self.y.clone(),
            self.y != -100,
            [["a0", "a1", "a2", "a3"], ["b0", "b1", "b2"]],
            self.lengths.clone(),
        )


def make_trainer():
    return Trainer(
        32,
        1,
        1,
        0.9,
        1e-3,
        3,
        "fixture",
        "test",
        0.5,
        input_dim=32,
        enc_heads=4,
        pred_heads=4,
        use_wandb=False,
        student_encoder_attention="block_causal",
        student_encoder_block_size=4,
        rope_mode="1d_flat",
    )


@pytest.mark.parametrize("head_mode", ["single", "foreground_activity"])
def test_fit_and_evaluate_saved_linear_head(tmp_path, head_mode):
    trainer = make_trainer()
    stats = trainer.run_linear_probe(
        TinyGenerator(),
        batch_size=2,
        device="cpu",
        val_batch_gen=TinyGenerator(),
        val_batch_size=2,
        probe_epochs=2,
        linear_only=True,
        linear_probe_pool=False,
        select_metric="lin_combined",
        linear_probe_head_mode=head_mode,
        background_label_id=0,
        lin_lr_milestones=None,
    )
    assert stats and "lin_combined" in stats
    path = tmp_path / "linear.pt"
    torch.save(trainer.lin_probe.state_dict(), path)
    score = evaluate_linear(
        trainer,
        TinyGenerator(),
        path,
        device="cpu",
        batch_size=2,
        linear_probe_pool=False,
        linear_probe_head_mode=head_mode,
        background_label_id=0,
    )
    assert score["frame_acc"] == pytest.approx(stats["lin_frame_acc"])
    assert score["combined"] == pytest.approx(stats["lin_combined"])


@pytest.mark.parametrize("kind", ["ltcontext", "causal_ltcontext"])
def test_fit_temporal_probe(tmp_path, kind):
    trainer = make_trainer()
    cfg = {
        "num_layers": 2,
        "num_stages": 2,
        "model_dim": 16,
        "windowed_attn_w": 4,
        "long_term_attn_g": 4,
        "num_attn_heads": 1,
        "attention_mode": "block_causal",
        "block_size": 4,
    }
    stats = trainer.run_linear_probe(
        TinyGenerator(),
        batch_size=2,
        device="cpu",
        val_batch_gen=TinyGenerator(),
        val_batch_size=2,
        probe_epochs=1,
        linear_only=False,
        temporal_probe_kind=kind,
        ltcontext_cfg_overrides=cfg,
        linear_probe_pool=False,
        select_metric="ltcontext_combined",
        lin_lr_milestones=None,
    )
    assert stats and "ltcontext_combined" in stats
    lin = tmp_path / "linear.pt"
    temp = tmp_path / "temporal.pt"
    torch.save(trainer.lin_probe.state_dict(), lin)
    torch.save(trainer.ltcontext_probe.state_dict(), temp)
    restored = trainer.evaluate_loaded_ltcontext_probes(
        TinyGenerator(),
        2,
        "cpu",
        str(temp),
        str(lin),
        temporal_probe_kind=kind,
        ltcontext_cfg_overrides=cfg,
        linear_probe_pool=False,
    )
    assert restored["ltcontext_combined"] == pytest.approx(stats["ltcontext_combined"])


def test_portable_config_roots(tmp_path, monkeypatch):
    from pjepa.utils.runtime import load_json

    monkeypatch.setenv("PJEPA_DATA_ROOT", str(tmp_path / "other_data"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"paths": {"archive": "{data_root}/features.pt"}}))
    assert load_json(path)["paths"]["archive"] == str(tmp_path / "other_data/features.pt")


def test_training_loop_saves_loadable_snapshot(tmp_path):
    trainer = make_trainer()
    trainer.train(
        str(tmp_path),
        TinyGenerator(),
        1,
        2,
        0.001,
        "cpu",
        val_batch_gen=TinyGenerator(),
        val_batch_size=2,
        probe_mode="linear",
        probe_every=1,
        probe_epochs=1,
        linear_probe_pool=False,
        ckpt_metric="lin_combined",
    )
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "last.pt").is_file()
    restored, metadata = load_model(tmp_path / "last.pt")
    x = TinyGenerator().x
    torch.testing.assert_close(encode(restored, x), encode(trainer.model.eval(), x), rtol=0, atol=0)


def test_head_encoder_identity_survives_move(tmp_path):
    from pjepa.checkpoints import assert_encoder_identity, sha256_file

    path = tmp_path / "renamed.pt"
    path.write_bytes(b"checkpoint fixture")
    metadata = {
        "pjepa_checkpoint": "/old/machine/model.pt",
        "pjepa_checkpoint_sha256": sha256_file(path),
    }
    assert_encoder_identity(metadata, path)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="do not match"):
        assert_encoder_identity(metadata, path)


def test_lemon_training_end_to_end(tmp_path):
    from pjepa.training.lemon import train

    n = 80
    tokens_per_video = 8
    dim = 32
    torch.save(
        {
            "dataset": "cholec80",
            "video_ids": [f"video{i:02d}" for i in range(1, n + 1)],
            "source_splits": ["train"] * 40 + ["test"] * 40,
            "tokens": torch.randn(n * tokens_per_video, dim),
            "offsets": torch.arange(n + 1) * tokens_per_video,
            "frame_indices": torch.arange(tokens_per_video).repeat(n),
            "phase_labels": torch.arange(tokens_per_video).repeat(n) % 3,
            "phase_class_names": ["a", "b", "c"],
            "feature_dim": dim,
            "target_fps": 1.0,
        },
        tmp_path / "phases.pt",
    )
    torch.save(
        {
            "tokens": torch.randn(24, dim),
            "times": torch.arange(24).float(),
            "offsets": torch.tensor([0, 12, 24]),
        },
        tmp_path / "shard.pt",
    )
    torch.save(
        {
            "shards": ["shard.pt"],
            "shard_ids": torch.tensor([0, 0]),
            "local_indices": torch.tensor([0, 1]),
            "video_ids": ["one", "two"],
            "paths": ["one", "two"],
            "splits": ["pretrain", "pretrain"],
            "num_tokens": torch.tensor([12, 12]),
            "feature_dim": dim,
            "target_fps": 1.0,
            "shard_root": "/stale/root",
        },
        tmp_path / "index.pt",
    )
    config = {
        "runtime": {"seed": 17, "device": "cpu"},
        "experiment": {"run_name": "fixture"},
        "paths": {
            "lemon_index": str(tmp_path / "index.pt"),
            "cholec80_archive": str(tmp_path / "phases.pt"),
            "ckpt_dir": str(tmp_path / "weights"),
        },
        "data": {"sample_rate": 1, "max_sequence_len": 16, "cholec80_protocol": "development"},
        "model": {
            "input_dim": dim,
            "dim": dim,
            "enc_depth": 1,
            "pred_depth": 1,
            "enc_heads": 4,
            "pred_heads": 4,
        },
        "ssl": {
            "student_encoder_attention": "block_causal",
            "student_encoder_block_size": 4,
            "rope_mode": "1d_flat",
            "mask_ratio": 0.5,
            "ema_momentum": 0.9,
        },
        "train": {"num_epochs": 1, "batch_size": 2, "lr": 0.001, "use_bfloat16": False},
        "validation": {"every": 1, "batch_size": 2},
        "probe": {
            "every": 1,
            "epochs": 1,
            "extraction_batch_size": 4,
            "token_batch_size": 32,
            "run_raw_baseline": False,
        },
        "loader": {"lemon": {"num_workers": 0}, "surgical_phase": {"num_workers": 0}},
        "wandb": {"use_wandb": False},
    }
    result = train(config)
    model, meta = load_model(result["best_checkpoint"])
    assert meta["architecture"]["enc_heads"] == 4
    assert torch.isfinite(encode(model, torch.randn(1, 9, dim))).all()


@pytest.mark.parametrize(
    "attention,rope", [("block_causal", "1d_flat"), ("clip_causal", "2d_clip_frame")]
)
def test_feature_runner_archive_train_fit_evaluate(tmp_path, attention, rope):
    """Exercise archive labels, snapshots, fitting, and scoring through the public runner."""
    import csv
    from pjepa.cli.feature_experiment import run

    archive = tmp_path / "features.pt"
    torch.save(
        {
            "paths": ["train/view.pt", "val/view.pt"],
            "tokens": torch.randn(24, 32),
            "times": torch.arange(12, dtype=torch.float32).repeat(2),
            "offsets": torch.tensor([0, 12, 24]),
        },
        archive,
    )
    manifest = tmp_path / "segments.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "video_path",
                "official_split",
                "action_type",
                "action_id",
                "start_sec",
                "end_sec",
            ],
        )
        writer.writeheader()
        for split in ("train", "val"):
            for label in range(3):
                writer.writerow(
                    dict(
                        video_path=f"{split}/view.mp4",
                        official_split=split,
                        action_type="assembly",
                        action_id=label,
                        start_sec=label * 4,
                        end_sec=(label + 1) * 4,
                    )
                )
    config = {
        "runtime": {"seed": 19, "device": "cpu"},
        "experiment": {"dataset": "assembly101", "split": "fixture", "run_name": "fixture"},
        "paths": {
            "labels_root": str(tmp_path),
            "features_root": str(tmp_path),
            "output_root": str(tmp_path / "outputs"),
        },
        "data": {
            "features_dim": 32,
            "input_features_dim": 32,
            "archive_path": str(archive),
            "train_manifest": manifest.name,
            "val_manifest": manifest.name,
            "label_column": "action_id",
            "target_fps": 1.0,
        },
        "loader": {"backend": "assembly101_frame_flat_archive"},
        "ssl": {
            "enc_depth": 1,
            "pred_depth": 1,
            "enc_heads": 4,
            "pred_heads": 4,
            "ema_momentum": 0.9,
            "mask_ratio": 0.5,
            "student_encoder_attention": attention,
            "student_encoder_block_size": 4,
            "rope_mode": rope,
        },
        "train": {"lr": 0.001, "batch_size": 1, "num_epochs": 1, "val_every": 1},
        "probe": {"mode": "linear", "epochs": 1, "every": 1},
        "evaluation": {"batch_size": 1, "val_batch_size": 1, "probe_feature_source": "student"},
    }
    run(config, action="train")
    output = tmp_path / "outputs/fixture"
    checkpoint = output / "last.pt"
    fitted = run(config, action="fit-probe", checkpoint=str(checkpoint))
    scored = run(
        config, action="evaluate", checkpoint=str(checkpoint), linear_head=str(output / "linear.pt")
    )
    assert fitted["lin_frame_acc"] == scored["frame_acc"]
    assert json.loads((output / "config.json").read_text())["ssl"]["rope_mode"] == rope
