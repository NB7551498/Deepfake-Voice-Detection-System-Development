import pytest
import requests
from src.api.url_downloader import is_safe_url

def test_ssrf_protection():
    # Loopback and private ranges must be blocked
    safe, msg = is_safe_url("http://127.0.0.1:8000/evil.wav")
    assert not safe
    assert "forbidden" in msg or "prohibited" in msg

    safe, msg = is_safe_url("http://localhost:8000/evil.wav")
    assert not safe

    safe, msg = is_safe_url("http://169.254.169.254/latest/meta-data")
    assert not safe
    assert "prohibited" in msg

    # Public web URL should pass security validation
    safe, msg = is_safe_url("https://example.com/test.wav")
    assert safe
    assert msg == ""

def test_api_predict_url_endpoint():
    url = "http://127.0.0.1:8001/predict-url"
    # Testing SSRF block from API
    resp = requests.post(url, json={"url": "http://127.0.0.1/test.wav", "mode": "fast"})
    assert resp.status_code == 400
    assert "forbidden" in resp.json()["detail"] or "prohibited" in resp.json()["detail"]
