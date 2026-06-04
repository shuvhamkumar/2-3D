"""Tests for mask resolution / naming / coverage (no binaries)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from engine.config import MaskingConfig
from engine.masks import (
    MaskError,
    colmap_mask_filename,
    openmvs_mask_filename,
    prepare_masks,
    resolve_mask_dir,
    write_openmvs_masks,
)


def _img(path, size=(64, 48)):
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def _mask(path, size=(64, 48)):
    Image.fromarray(np.full((size[1], size[0]), 255, dtype=np.uint8), mode="L").save(path)


def test_naming_conventions():
    assert colmap_mask_filename("frame001.JPG") == "frame001.JPG.png"
    assert openmvs_mask_filename("frame001.JPG") == "frame001.mask.png"
    assert openmvs_mask_filename("P1.jpeg") == "P1.mask.png"


def test_resolve_mask_dir_explicit(tmp_path):
    cfg = MaskingConfig(mask_dir=tmp_path / "m")
    d, src = resolve_mask_dir(tmp_path / "imgs", tmp_path / "job", cfg)
    assert d == tmp_path / "m" and src == "external"


def test_resolve_mask_dir_auto(tmp_path):
    cfg = MaskingConfig(auto_mask=True)
    d, src = resolve_mask_dir(tmp_path / "imgs", tmp_path / "job", cfg)
    assert d == tmp_path / "job" / "masks" and src == "auto(rembg)"


def test_resolve_mask_dir_sibling(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    (tmp_path / "masks").mkdir()
    d, src = resolve_mask_dir(imgs, tmp_path / "job", MaskingConfig())
    assert d == tmp_path / "masks" and src == "sibling"


def test_resolve_mask_dir_none(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    d, src = resolve_mask_dir(imgs, tmp_path / "job", MaskingConfig())
    assert d is None and src == "none"


def test_prepare_masks_full_coverage(tmp_path):
    imgs = tmp_path / "imgs"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()
    for i in range(5):
        _img(imgs / f"f{i}.png")
        _mask(masks / f"f{i}.png.png")  # COLMAP naming: <image_name>.png
    info = prepare_masks(imgs, tmp_path / "job", MaskingConfig(mask_dir=masks))
    assert info is not None
    assert info.n_images == 5 and info.n_masks == 5
    assert info.coverage == 1.0


def test_prepare_masks_partial_coverage_warns(tmp_path):
    imgs = tmp_path / "imgs"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()
    for i in range(4):
        _img(imgs / f"f{i}.png")
    _mask(masks / "f0.png.png")
    _mask(masks / "f1.png.png")
    info = prepare_masks(imgs, tmp_path / "job", MaskingConfig(mask_dir=masks))
    assert info.n_masks == 2 and info.coverage == 0.5


def test_prepare_masks_none_when_no_masks(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _img(imgs / "f0.png")
    assert prepare_masks(imgs, tmp_path / "job", MaskingConfig()) is None


def test_prepare_masks_errors_on_empty_mask_dir(tmp_path):
    imgs = tmp_path / "imgs"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()
    _img(imgs / "f0.png")
    with pytest.raises(MaskError, match="No masks found"):
        prepare_masks(imgs, tmp_path / "job", MaskingConfig(mask_dir=masks))


def test_write_openmvs_masks(tmp_path):
    masks = tmp_path / "masks"
    dense_imgs = tmp_path / "dense" / "images"
    masks.mkdir()
    dense_imgs.mkdir(parents=True)
    for i in range(3):
        _img(dense_imgs / f"f{i}.JPG")
        _mask(masks / f"f{i}.JPG.png")
    n = write_openmvs_masks(masks, dense_imgs)
    assert n == 3
    assert (dense_imgs / "f0.mask.png").exists()
    assert (dense_imgs / "f2.mask.png").exists()


def test_masking_config_defaults():
    cfg = MaskingConfig()
    assert cfg.bbox_crop is True
    assert cfg.bbox_margin == 0.15
    assert cfg.mask_dir is None and cfg.auto_mask is False
    assert cfg.mask_dilate_frac == 0.0


def test_dilate_mask_grows_foreground():
    from engine.masks import dilate_mask

    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[48:52, 48:52] = 255  # small 4x4 foreground square
    before = (arr > 127).sum()
    grown = dilate_mask(arr, px=5)
    after = (grown > 127).sum()
    assert after > before
    # dilating by 0 is a no-op (just re-binarised)
    assert (dilate_mask(arr, 0) > 127).sum() == before


def test_external_mask_dilation_materialises(tmp_path):
    imgs = tmp_path / "imgs"
    masks = tmp_path / "masks"
    job = tmp_path / "job"
    imgs.mkdir()
    masks.mkdir()
    for i in range(3):
        _img(imgs / f"f{i}.png", size=(120, 120))
        m = np.zeros((120, 120), dtype=np.uint8)
        m[55:65, 55:65] = 255
        Image.fromarray(m, "L").save(masks / f"f{i}.png.png")
    info = prepare_masks(imgs, job, MaskingConfig(mask_dir=masks, mask_dilate_frac=0.1))
    assert "dilated" in info.source
    assert (job / "masks_dilated").is_dir()
    # the dilated mask has more foreground than the original
    orig = np.asarray(Image.open(masks / "f0.png.png").convert("L")) > 127
    dil = np.asarray(Image.open(job / "masks_dilated" / "f0.png.png").convert("L")) > 127
    assert dil.sum() > orig.sum()
