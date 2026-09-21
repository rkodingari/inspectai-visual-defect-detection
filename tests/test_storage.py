from types import SimpleNamespace

from inspectai.storage import PredictionStore


def test_prediction_and_feedback_round_trip(tmp_path):
    store = PredictionStore(tmp_path / "test.db")
    result = SimpleNamespace(label="normal", confidence=0.8, latency_ms=2.5,
                             model_version="test-v1", probabilities={"normal": 0.8, "defective": 0.2})
    prediction_id = store.add("sample.png", result)
    assert store.feedback(prediction_id, "correct")
    assert store.recent()[0]["feedback"] == "correct"

