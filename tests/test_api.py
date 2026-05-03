"""
API endpoint integration tests.
Run: pytest tests/test_api.py -v

These test the FastAPI app directly using httpx TestClient.
No need for a running server.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "name" in data
        assert data["status"] == "running"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_docs_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


class TestInvoiceEndpoints:
    def test_platforms_list(self, client):
        r = client.get("/api/v2/invoice/platforms")
        assert r.status_code == 200
        data = r.json()
        assert "amazon" in data["supported_platforms"]
        assert "flipkart" in data["supported_platforms"]
        assert len(data["supported_platforms"]) >= 9

    def test_extract_rejects_unsupported_type(self, client):
        r = client.post(
            "/api/v2/invoice/extract",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert r.status_code == 400
        assert "Unsupported" in r.json()["detail"]


class TestExportEndpoint:
    def test_csv_export(self, client):
        invoices = [
            {
                "platform": "amazon",
                "invoice_number": "IN-123",
                "total_amount": 1294.46,
                "currency": "INR",
                "fields_extracted": 5,
                "fields_total": 16,
                "validation_warnings": [],
            }
        ]
        r = client.post("/api/v2/export/csv", json=invoices)
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        assert "Platform" in r.text  # Header row
        assert "amazon" in r.text
