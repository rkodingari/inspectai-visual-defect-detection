from inspectai.data.dataset import BinaryDefectDataset, validate_dataset


def test_synthetic_dataset_is_valid(synthetic_data):
    counts = validate_dataset(synthetic_data)
    assert counts["train"] == {"normal": 24, "defective": 24}
    dataset = BinaryDefectDataset(synthetic_data, "test")
    image, label, path = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label in (0, 1)
    assert path.endswith(".png")


def test_validation_rejects_missing_classes(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Missing images"):
        validate_dataset(tmp_path)

