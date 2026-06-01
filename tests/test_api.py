"""
NIDS REST API serving unit tests
================================
Tests endpoints under valid, batch, and malformed request payload schemas
using fastapi.testclient.TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# Instantiate test client within context lifespan to trigger model loading
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Sample valid connection schema (tcp connection to http SF port)
VALID_RECORD = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 215,
    "dst_bytes": 4507,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_hot_login": 0,
    "is_guest_login": 0,
    "count": 1,
    "srv_count": 1,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 51,
    "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.02,
    "dst_host_srv_diff_host_rate": 0.05,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

def test_read_root(client):
    """Test that GET / returns metadata and online state."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "version" in data
    assert data["status"] == "online"


def test_health_check(client):
    """Test that GET /health returns status healthy and caches are loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_predict_single(client):
    """Test that POST /predict returns correct schema validations."""
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 200
    data = response.json()
    
    # Assert validation contracts
    assert isinstance(data["is_attack"], bool)
    assert isinstance(data["attack_family"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["stage1_score"], float)
    assert data["attack_family"] in ["normal", "dos", "probe", "r2l", "u2r"]


def test_predict_batch(client):
    """Test that POST /predict_batch scores multiple logs successfully."""
    batch_payload = [VALID_RECORD, VALID_RECORD, VALID_RECORD]
    response = client.post("/predict_batch", json=batch_payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    
    for pred in data["predictions"]:
        assert isinstance(pred["is_attack"], bool)
        assert isinstance(pred["attack_family"], str)
        assert isinstance(pred["confidence"], float)
        assert isinstance(pred["stage1_score"], float)


def test_malformed_pydantic_schema(client):
    """Test that submitting an incomplete record yields a 422 validation error."""
    # Omit duration and src_bytes keys
    malformed_record = VALID_RECORD.copy()
    del malformed_record["duration"]
    del malformed_record["src_bytes"]
    
    response = client.post("/predict", json=malformed_record)
    assert response.status_code == 422  # Pydantic schema validation error


def test_empty_batch(client):
    """Test that POST /predict_batch with an empty array returns empty list."""
    response = client.post("/predict_batch", json=[])
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == []
    assert data["count"] == 0
