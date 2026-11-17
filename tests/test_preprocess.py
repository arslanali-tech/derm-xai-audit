"""Images and masks must be resized with identical geometry, and masks must stay binary."""

import numpy as np
import pytest

from dermxai.data.preprocess import read_gray, read_image_bgr, resize_pair, resized_shape, write_png_atomic


def test_resized_shape_ham10000() -> None:
    assert resized_shape(450, 600, 256) == (256, 341)  # landscape 600x450 original
    assert resized_shape(600, 450, 256) == (341, 256)  # portrait keeps aspect ratio too


def toy_pair() -> tuple[np.ndarray, np.ndarray]:
    """600x450 noisy image with a known rectangular 'lesion' in the mask."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (450, 600, 3), dtype=np.uint8)
    mask = np.zeros((450, 600), dtype=np.uint8)
    mask[90:270, 150:390] = 255  # rows 20%-60%, cols 25%-65%
    return img, mask


def test_resize_pair_same_size_and_binary_mask() -> None:
    img, mask = toy_pair()
    img_s, mask_s = resize_pair(img, mask, 256)
    assert img_s.shape == (256, 341, 3)
    assert mask_s.shape == (256, 341)
    assert set(np.unique(mask_s)) <= {0, 255}


def test_mask_region_scales_correctly() -> None:
    img, mask = toy_pair()
    _, mask_s = resize_pair(img, mask, 256)
    expected = np.zeros((256, 341), dtype=bool)
    expected[round(0.2 * 256):round(0.6 * 256), round(0.25 * 341):round(0.65 * 341)] = True
    got = mask_s > 0
    iou = (got & expected).sum() / (got | expected).sum()
    assert iou > 0.97
    assert got.mean() == pytest.approx(mask.astype(bool).mean(), abs=0.01)  # area fraction preserved


def test_soft_mask_is_binarised() -> None:
    img, mask = toy_pair()
    mask = (mask // 2).astype(np.uint8)  # values 0 / 127: below the >127 threshold -> empty
    mask[0:10, 0:10] = 200
    _, mask_s = resize_pair(img, mask, 256)
    assert set(np.unique(mask_s)) <= {0, 255}


def test_mismatched_sizes_raise() -> None:
    img, mask = toy_pair()
    with pytest.raises(ValueError):
        resize_pair(img, mask[:-1], 256)


def test_png_roundtrip_is_lossless(tmp_path) -> None:
    img, mask = toy_pair()
    write_png_atomic(tmp_path / "img.png", img)
    write_png_atomic(tmp_path / "mask.png", mask)
    assert np.array_equal(read_image_bgr(tmp_path / "img.png"), img)
    assert np.array_equal(read_gray(tmp_path / "mask.png"), mask)
    assert not list(tmp_path.glob("*.tmp"))  # no temp files left behind
