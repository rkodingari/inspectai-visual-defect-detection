import torch

from inspectai.explainability.gradcam import GradCAM
from inspectai.models.factory import create_model, trainable_parameters


def test_baseline_forward_and_gradcam():
    model = create_model("baseline_cnn")
    inputs = torch.randn(1, 3, 64, 64)
    assert model(inputs).shape == (1, 2)
    cam = GradCAM(model, model.gradcam_layer)
    heatmap = cam(inputs)
    cam.close()
    assert heatmap.shape == (64, 64)
    assert 0 <= heatmap.min() <= heatmap.max() <= 1


def test_mobilenet_freezes_most_parameters():
    model = create_model("mobilenet_v3_small", pretrained=False, freeze_backbone=True)
    trainable, total = trainable_parameters(model)
    assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2)
    assert trainable < total / 2


def test_additional_classifiers_forward_and_freeze():
    for name in ("efficientnet_b0", "resnet18"):
        model = create_model(name, pretrained=False, freeze_backbone=True).eval()
        trainable, total = trainable_parameters(model)
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 2)
        assert trainable < total
