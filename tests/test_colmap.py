"""Tests for COLMAP output parsing / DB reading and matcher selection."""

from __future__ import annotations

import sqlite3

from engine.binaries import Environment
from engine.colmap import parse_model_analyzer, read_database_stats
from engine.config import MatcherStrategy, Preset, UseCase, build_preset
from engine.stages.base import StageContext
from engine.stages.matching import select_strategy
from engine.workspace import Workspace

# Real COLMAP 3.9 model_analyzer output (glog-prefixed lines).
_ANALYZER_OUTPUT = """
I20260531 cameras Cameras: 1
I20260531 Images: 128
I20260531 Registered images: 126
I20260531 Points: 63215
I20260531 Observations: 308913
I20260531 Mean track length: 4.8867
I20260531 Mean observations per image: 2451.69
I20260531 Mean reprojection error: 0.6712px
"""


def test_parse_model_analyzer():
    m = parse_model_analyzer(_ANALYZER_OUTPUT)
    assert m["cameras"] == 1
    assert m["images"] == 128
    assert m["registered_images"] == 126
    assert m["points"] == 63215
    assert abs(m["mean_track_length"] - 4.8867) < 1e-6
    assert abs(m["mean_reproj_error"] - 0.6712) < 1e-6


def test_parse_model_analyzer_missing_fields_are_omitted():
    m = parse_model_analyzer("Images: 5")
    assert m == {"images": 5}


def _make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
    con.executemany("INSERT INTO images (image_id) VALUES (?)", [(i,) for i in range(1, 6)])
    con.executemany(
        "INSERT INTO keypoints (image_id, rows, cols) VALUES (?, ?, ?)",
        [(i, 100 * i, 4) for i in range(1, 6)],
    )
    con.executemany(
        "INSERT INTO two_view_geometries (pair_id, rows) VALUES (?, ?)",
        [(1, 50), (2, 0), (3, 30)],  # one pair has 0 verified matches
    )
    con.commit()
    con.close()


def test_read_database_stats(tmp_path):
    db = tmp_path / "database.db"
    _make_db(db)
    stats = read_database_stats(db)
    assert stats.num_images == 5
    assert stats.num_descriptors_images == 5
    assert stats.num_keypoints == 100 + 200 + 300 + 400 + 500
    assert stats.num_matched_pairs == 2  # the 0-row pair excluded
    assert stats.num_inlier_matches == 80


def test_read_database_stats_missing_tables(tmp_path):
    db = tmp_path / "empty.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()
    stats = read_database_stats(db)
    assert stats.num_images == 0
    assert stats.num_matched_pairs == 0


def _ctx(strategy=None, vocab=None, exhaustive_max=150):
    cfg = build_preset(Preset.DRAFT, UseCase.OBJECT)
    if strategy is not None:
        cfg.matching.strategy = strategy
    cfg.matching.vocab_tree_path = vocab
    cfg.matching.exhaustive_max_images = exhaustive_max
    ws = Workspace(root=__import__("pathlib").Path("/tmp/x"), image_dir=__import__("pathlib").Path("/tmp/y"))
    return StageContext(workspace=ws, config=cfg, env=Environment())


def test_select_strategy_respects_explicit():
    ctx = _ctx(strategy=MatcherStrategy.SEQUENTIAL)
    assert select_strategy(ctx, 1000) is MatcherStrategy.SEQUENTIAL


def test_select_strategy_auto_small_is_exhaustive():
    ctx = _ctx(strategy=MatcherStrategy.AUTO)
    assert select_strategy(ctx, 128) is MatcherStrategy.EXHAUSTIVE


def test_select_strategy_auto_large_uses_vocab_when_available(tmp_path):
    tree = tmp_path / "tree.bin"
    tree.write_text("x")
    ctx = _ctx(strategy=MatcherStrategy.AUTO, vocab=tree, exhaustive_max=100)
    assert select_strategy(ctx, 500) is MatcherStrategy.VOCAB_TREE


def test_select_strategy_auto_large_falls_back_to_exhaustive():
    ctx = _ctx(strategy=MatcherStrategy.AUTO, exhaustive_max=100)
    assert select_strategy(ctx, 500) is MatcherStrategy.EXHAUSTIVE
