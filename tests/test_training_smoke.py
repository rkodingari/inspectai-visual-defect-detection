from inspectai.training.train import TrainConfig, train


def test_one_epoch_training_smoke(synthetic_data, tmp_path):
    checkpoint = train(TrainConfig(
        data_dir=str(synthetic_data), model="baseline_cnn", epochs=1,
        batch_size=16, image_size=64, output_dir=str(tmp_path / "artifacts"),
        pretrained=False,
    ))
    assert checkpoint.exists()

