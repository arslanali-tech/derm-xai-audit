"""Only real image files are extracted from the dataset zips."""

from dermxai.data.download import is_data_member


def test_real_images_are_kept() -> None:
    assert is_data_member("ISIC_0024306.jpg")
    assert is_data_member("HAM10000_segmentations_lesion_tschandl/ISIC_0024306_segmentation.png")


def test_macos_junk_and_other_files_are_skipped() -> None:
    assert not is_data_member("HAM10000_segmentations_lesion_tschandl/._ISIC_0025504_segmentation.png")
    assert not is_data_member("__MACOSX/masks/ISIC_0025504_segmentation.png")
    assert not is_data_member("readme.txt")
    assert not is_data_member("folder/")
