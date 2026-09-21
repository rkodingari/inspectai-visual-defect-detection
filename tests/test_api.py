from fastapi.testclient import TestClient

from inspectai.api.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint():
    response = TestClient(app).get("/models")
    assert response.status_code == 200
    assert "models" in response.json()


def test_rejects_non_image_upload():
    response = TestClient(app).post(
        "/predict", files={"file": ("bad.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415
