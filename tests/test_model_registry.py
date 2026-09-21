import json

import pytest

import inspectai.model_registry as registry_module


def test_registry_lists_and_resolves_existing_models(tmp_path, monkeypatch):
    default = tmp_path / "best_model.pt"
    default.touch()
    mobile = tmp_path / "mobilenet.pt"
    mobile.touch()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "best_model": "mobilenet_v3_small",
        "models": {
            "mobilenet_v3_small": {
                "checkpoint": str(mobile), "display_name": "MobileNetV3-Small",
                "best_val_f1": 0.8,
            }
        },
    }))
    monkeypatch.setattr(registry_module, "DEFAULT_CHECKPOINT", default)
    available = registry_module.available_models(registry)
    assert set(available) == {"best", "mobilenet_v3_small"}
    assert registry_module.resolve_checkpoint("mobilenet_v3_small", registry) == mobile
    with pytest.raises(ValueError, match="unavailable"):
        registry_module.resolve_checkpoint("resnet18", registry)
