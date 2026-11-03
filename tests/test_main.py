from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    assert client.get("/").status_code == 200

def test_health():
    assert client.get("/health").status_code == 200

def test_heartbeat():
    assert client.get("/heartbeat").status_code == 200

def test_tools():
    assert client.get("/tools").status_code == 200

def test_presence_current():
    response = client.get("/v1/presence/current")
    assert response.status_code == 200
    assert "presence_score" in response.json()

def test_presence_location():
    response = client.get("/v1/presence/location?lat=3.139&lon=101.6869")
    assert response.status_code == 200
    assert "presence_score" in response.json()

def test_activity():
    response = client.get("/v1/activity?lat=3.139&lon=101.6869")
    assert response.status_code == 200
    assert "activity_score" in response.json()

def test_sleep():
    response = client.get("/v1/sleep?lat=3.139&lon=101.6869")
    assert response.status_code == 200
    assert "awake_probability" in response.json()

def test_darkness():
    response = client.get("/v1/darkness?lat=3.139&lon=101.6869")
    assert response.status_code == 200
    assert "darkness_score" in response.json()

def test_calendar():
    response = client.get("/v1/calendar?lat=3.139&lon=101.6869")
    assert response.status_code == 200
    assert "calendar_state" in response.json()
