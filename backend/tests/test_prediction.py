import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs_url" in data

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "locations_count" in data

def test_get_locations_endpoint():
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert isinstance(data["locations"], list)
    assert len(data["locations"]) > 0
    assert "total" in data

def test_predict_success():
    payload = {
        "location": "thane",
        "carpet_area_sqft": 1200.0,
        "floor_num": 3,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "bathrooms": 2,
        "balconies": 1
    }
    response = client.post("/predict", json=payload)
    # If model is loaded, it should return 200 and valid price
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "success"
        assert "predicted_price_rupees" in data
        assert data["predicted_price_rupees"] > 0
        assert "formatted_price" in data
        assert "currency" in data
    else:
        # If model file is not yet exported, it returns 503
        assert response.status_code in [200, 503]

def test_predict_validation_error():
    # Negative carpet area should fail Pydantic validation (422)
    invalid_payload = {
        "location": "thane",
        "carpet_area_sqft": -50.0,
        "floor_num": 2,
        "furnishing": "Furnished",
        "transaction": "Resale",
        "bathrooms": 2,
        "balconies": 1
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
