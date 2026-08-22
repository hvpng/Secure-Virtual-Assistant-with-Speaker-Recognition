from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from module_a.src import trainer as trainer_module
from module_a.src.checkpoint import CheckpointError, load_checkpoint
from module_a.src.config import ConfigurationError, load_config
from module_a.src.model_factory import build_model, build_optimizer, build_scheduler
from module_a.src.models.wavlm_frontend import DeterministicFakeWavLM
from module_a.src.trainer import (
    TrainerError,
    create_grad_scaler,
    prepare_training_data,
    train_model,
)


class FakeGradScaler:
    """CPU test double with GradScaler-compatible overflow semantics."""

    def __init__(self, overflows: list[bool], *, initial_scale: float = 8.0):
        self.overflows = list(overflows)
        self.current_scale = initial_scale
        self.current_overflow = False
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0
        self.optimizer_step_calls = 0

    def is_enabled(self):
        return True

    def scale(self, loss):
        self.scale_calls += 1
        return loss

    def unscale_(self, optimizer):
        del optimizer
        self.unscale_calls += 1

    def get_scale(self):
        return self.current_scale

    def step(self, optimizer):
        self.step_calls += 1
        self.current_overflow = self.overflows.pop(0) if self.overflows else False
        if not self.current_overflow:
            optimizer.step()
            self.optimizer_step_calls += 1

    def update(self):
        self.update_calls += 1
        if self.current_overflow:
            self.current_scale *= 0.5

    def state_dict(self):
        return {"scale": self.current_scale}

    def load_state_dict(self, state):
        self.current_scale = float(state["scale"])


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def state_dict(self):
        return {"step_calls": self.step_calls}

    def load_state_dict(self, state):
        self.step_calls = int(state["step_calls"])


def _history_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _install_fake_amp(monkeypatch, scaler, scheduler):
    monkeypatch.setattr(
        trainer_module,
        "create_grad_scaler",
        lambda _device, _enabled: scaler,
    )
    monkeypatch.setattr(
        trainer_module,
        "build_scheduler",
        lambda *_args, **_kwargs: scheduler,
    )


def build_training_fixture(tmp_path, write_wav, small_model_config, *, max_steps=2):
    dataset_root = tmp_path / "dataset"
    manifest = tmp_path / "train_manifest.csv"
    rows = []
    for speaker_index in range(4):
        for utterance_index in range(4):
            relative = Path(f"speaker_{speaker_index}") / f"utt_{utterance_index}.wav"
            write_wav(dataset_root / relative, duration_sec=0.1)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "speaker_id": f"speaker_{speaker_index}",
                    "split": "train",
                }
            )
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "speaker_id", "split"))
        writer.writeheader()
        writer.writerows(rows)
    config = replace(
        small_model_config.with_overrides(
            dataset_root=dataset_root, output_root=tmp_path / "run"
        ),
        audio=replace(small_model_config.audio, segment_seconds=0.05),
        training=replace(
            small_model_config.training,
            epochs=1,
            max_steps=max_steps,
            mixed_precision=True,
            speakers_per_batch=2,
            utterances_per_speaker=2,
            num_workers=0,
            log_every_steps=1,
            max_train_speakers=4,
            max_monitor_speakers=2,
            monitor_holdout_ratio=0.25,
        ),
    )
    data = prepare_training_data(
        config, train_manifest=manifest, dataset_root=dataset_root
    )
    return config, data


def new_model(config, num_classes):
    return build_model(
        config,
        num_classes=num_classes,
        frontend=DeterministicFakeWavLM(768, frame_count=12),
    )


def test_cpu_one_step_changes_backend_not_frozen_frontend_and_writes_outputs(
    tmp_path, write_wav, small_model_config, capsys
):
    config, data = build_training_fixture(
        tmp_path, write_wav, small_model_config, max_steps=1
    )
    model = new_model(config, len(data.speaker_to_index))
    backend_before = model.encoder.projection.weight.detach().clone()
    frontend_before = {
        name: parameter.detach().clone()
        for name, parameter in model.encoder.frontend.named_parameters()
    }

    result = train_model(
        model=model,
        config=config,
        data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        debug_first_step=True,
    )
    captured = capsys.readouterr().out

    assert result.global_step == 1
    assert math.isfinite(result.final_train_loss)
    assert math.isfinite(result.final_monitor_loss)
    assert not torch.equal(backend_before, model.encoder.projection.weight)
    assert all(
        torch.equal(frontend_before[name], parameter)
        for name, parameter in model.encoder.frontend.named_parameters()
    )
    assert create_grad_scaler(torch.device("cpu"), True).is_enabled() is False
    assert "numerical_diagnostics_before_backward" in captured
    assert "campp_pool_variance_before_clamp" in captured
    assert "aam_sine_squared_before_clamp" in captured
    for filename in (
        "history.jsonl",
        "run_config.json",
        "speaker_to_index.json",
        "train_monitor_split.json",
        "training_summary.json",
    ):
        assert (tmp_path / "run" / filename).is_file()
    assert Path(result.checkpoint_path).is_file()


def test_fp32_non_finite_gradient_remains_fatal(
    tmp_path, write_wav, small_model_config
):
    config, data = build_training_fixture(
        tmp_path, write_wav, small_model_config, max_steps=1
    )
    config = replace(
        config,
        training=replace(config.training, mixed_precision=False),
    )
    model = new_model(config, len(data.speaker_to_index))
    model.encoder.layer_norm.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )

    with pytest.raises(
        TrainerError, match=r"Non-finite gradient.*encoder\.layer_norm\.weight"
    ):
        train_model(
            model=model,
            config=config,
            data=data,
            device=torch.device("cpu"),
            output_dir=tmp_path / "run",
        )


def test_amp_overflow_skips_update_and_scheduler_then_recovers(
    tmp_path, write_wav, small_model_config, monkeypatch
):
    config, data = build_training_fixture(
        tmp_path, write_wav, small_model_config, max_steps=1
    )
    model = new_model(config, len(data.speaker_to_index))
    scaler = FakeGradScaler([True, False], initial_scale=8.0)
    scheduler = CountingScheduler()
    _install_fake_amp(monkeypatch, scaler, scheduler)

    result = train_model(
        model=model,
        config=config,
        data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
    )

    events = _history_events(tmp_path / "run" / "history.jsonl")
    overflow = next(event for event in events if event["type"] == "amp_overflow")
    assert overflow["step"] == 0
    assert overflow["pending_step"] == 1
    assert overflow["old_scale"] == 8.0
    assert overflow["new_scale"] == 4.0
    assert result.global_step == 1
    assert scaler.step_calls == 2
    assert scaler.optimizer_step_calls == 1
    assert scheduler.step_calls == 1
    assert any(event["type"] == "train" and event["step"] == 1 for event in events)


def test_amp_gradient_accumulation_updates_only_at_boundary(
    tmp_path, write_wav, small_model_config, monkeypatch
):
    config, data = build_training_fixture(
        tmp_path, write_wav, small_model_config, max_steps=1
    )
    config = replace(
        config,
        training=replace(config.training, gradient_accumulation_steps=2),
    )
    scaler = FakeGradScaler([False])
    scheduler = CountingScheduler()
    _install_fake_amp(monkeypatch, scaler, scheduler)

    result = train_model(
        model=new_model(config, len(data.speaker_to_index)),
        config=config,
        data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
    )

    assert result.global_step == 1
    assert scaler.scale_calls == 2
    assert scaler.unscale_calls == 1
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1
    assert scaler.optimizer_step_calls == 1
    assert scheduler.step_calls == 1


def test_excessive_consecutive_amp_overflows_fail_controlled(
    tmp_path, write_wav, small_model_config, monkeypatch
):
    config, data = build_training_fixture(
        tmp_path, write_wav, small_model_config, max_steps=1
    )
    config = replace(
        config,
        training=replace(config.training, max_consecutive_amp_overflows=2),
    )
    scaler = FakeGradScaler([True, True], initial_scale=8.0)
    scheduler = CountingScheduler()
    _install_fake_amp(monkeypatch, scaler, scheduler)

    with pytest.raises(TrainerError, match="2 consecutive skipped updates.*scale 2.0"):
        train_model(
            model=new_model(config, len(data.speaker_to_index)),
            config=config,
            data=data,
            device=torch.device("cpu"),
            output_dir=tmp_path / "run",
        )

    events = _history_events(tmp_path / "run" / "history.jsonl")
    overflows = [event for event in events if event["type"] == "amp_overflow"]
    assert len(overflows) == 2
    assert [event["step"] for event in overflows] == [0, 0]
    assert scaler.optimizer_step_calls == 0
    assert scheduler.step_calls == 0


def test_resume_restores_step_optimizer_scheduler_and_rejects_mapping_mismatch(
    tmp_path, write_wav, small_model_config
):
    config, data = build_training_fixture(tmp_path, write_wav, small_model_config)
    first_model = new_model(config, len(data.speaker_to_index))
    first = train_model(
        model=first_model,
        config=config,
        data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        stop_after_steps=1,
        detect_anomaly=True,
    )
    restored_model = new_model(config, len(data.speaker_to_index))
    resumed = train_model(
        model=restored_model,
        config=config,
        data=data,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        resume=first.checkpoint_path,
    )

    assert first.global_step == 1
    assert resumed.global_step == 2
    assert resumed.resumed_from == first.checkpoint_path
    assert math.isfinite(resumed.final_train_loss)

    incompatible = dict(data.speaker_to_index)
    first_key, second_key = list(incompatible)[:2]
    incompatible[first_key], incompatible[second_key] = (
        incompatible[second_key],
        incompatible[first_key],
    )
    check_model = new_model(config, len(data.speaker_to_index))
    optimizer = build_optimizer(check_model, 1e-3)
    scheduler = build_scheduler(
        optimizer, scheduler_type="cosine", total_steps=2, warmup_steps=0
    )
    scaler = create_grad_scaler(torch.device("cpu"), False)
    with pytest.raises(CheckpointError, match="speaker_to_index"):
        load_checkpoint(
            first.checkpoint_path,
            model=check_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_num_classes=len(incompatible),
            expected_speaker_to_index=incompatible,
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("max_steps", "0"),
        ("speakers_per_batch", "0"),
        ("utterances_per_speaker", "0"),
        ("max_consecutive_amp_overflows", "0"),
    ],
)
def test_a3_config_rejects_non_positive_training_values(tmp_path, field, bad_value):
    source = Path("module_a/configs/experiment.yaml").read_text(encoding="utf-8")
    marker = f"  {field}: "
    lines = [
        f"{marker}{bad_value}" if line.startswith(marker) else line
        for line in source.splitlines()
    ]
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=field):
        load_config(experiment_config_path=experiment)


def test_train_cli_help_is_offline_and_exposes_safety_limits():
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "module_a.scripts.train_model", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "--mini" in result.stdout
    assert "--max-steps" in result.stdout
    assert "--resume" in result.stdout
    assert "--detect-anomaly" in result.stdout
    assert "--debug-first-step" in result.stdout
