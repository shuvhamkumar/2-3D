"""Tests for config parsing, presets, and file loading."""

from __future__ import annotations

import json

import pytest

from engine.config import (
    EngineConfig,
    MatcherStrategy,
    Preset,
    UseCase,
    build_preset,
)


def test_default_config_is_object_usecase_glb():
    cfg = EngineConfig()
    assert cfg.use_case is UseCase.OBJECT
    assert cfg.export.format.value == "glb"


@pytest.mark.parametrize("preset", list(Preset))
def test_every_preset_builds(preset):
    cfg = build_preset(preset)
    assert cfg.preset is preset
    # Quality monotonicity: draft <= balanced <= high on image size.
    assert cfg.feature.max_image_size > 0


def test_preset_quality_ordering():
    draft = build_preset(Preset.DRAFT)
    balanced = build_preset(Preset.BALANCED)
    high = build_preset(Preset.HIGH)
    sizes = [draft.feature.max_image_size, balanced.feature.max_image_size, high.feature.max_image_size]
    assert sizes == sorted(sizes)
    assert draft.mesh.refine is False
    assert high.mesh.refine is True


def test_object_usecase_selects_sequential_matcher():
    cfg = build_preset(Preset.BALANCED, UseCase.OBJECT)
    assert cfg.matching.strategy is MatcherStrategy.SEQUENTIAL
    assert cfg.feature.single_camera is True


def test_outdoor_usecase_requires_exif_and_spatial():
    cfg = build_preset(Preset.BALANCED, UseCase.OUTDOOR)
    assert cfg.matching.strategy is MatcherStrategy.SPATIAL
    assert cfg.ingest.require_exif_focal is True


def test_web_usecase_sets_decimation_budget():
    cfg = build_preset(Preset.BALANCED, UseCase.WEB)
    assert cfg.export.target_faces == 100_000


def test_from_dict_applies_preset_then_overrides():
    cfg = EngineConfig.from_dict(
        {"preset": "draft", "feature": {"max_num_features": 1234}}
    )
    # override wins
    assert cfg.feature.max_num_features == 1234
    # untouched preset value survives the deep-merge
    assert cfg.feature.max_image_size == build_preset(Preset.DRAFT).feature.max_image_size


def test_from_dict_without_preset_uses_model_defaults():
    cfg = EngineConfig.from_dict({"export": {"format": "obj"}})
    assert cfg.export.format.value == "obj"


def test_extra_keys_are_rejected():
    with pytest.raises(Exception):
        EngineConfig.model_validate({"not_a_field": 1})


def test_roundtrip_to_dict_is_revalidatable():
    cfg = build_preset(Preset.HIGH, UseCase.OUTDOOR)
    again = EngineConfig.model_validate(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_from_file_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("preset: high\nmvs:\n  resolution_level: 2\n", encoding="utf-8")
    cfg = EngineConfig.from_file(p)
    assert cfg.preset is Preset.HIGH
    assert cfg.mvs.resolution_level == 2  # override
    assert cfg.mesh.refine is True  # from high preset


def test_from_file_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"preset": "draft", "export": {"format": "ply"}}), encoding="utf-8")
    cfg = EngineConfig.from_file(p)
    assert cfg.export.format.value == "ply"
    assert cfg.feature.max_image_size == build_preset(Preset.DRAFT).feature.max_image_size


def test_from_file_bad_extension(tmp_path):
    p = tmp_path / "cfg.txt"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        EngineConfig.from_file(p)
