"""
tests/test_monitoring.py
=========================
Smoke tests for the Module 8 monitoring stack.
Verifies all components are running and metrics are flowing correctly.

Run:
    # After starting the stack (make monitoring-up):
    pytest tests/test_monitoring.py -v

    # Or directly:
    python tests/test_monitoring.py

    # Skip slow tests:
    pytest tests/test_monitoring.py -v -m "not slow"
"""

import json
import time
import sys
import os

import pytest
import requests

# ── Config ───────────────────────────────────────────────────
API_URL          = os.getenv("API_URL",          "http://localhost:8000")
PROMETHEUS_URL   = os.getenv("PROMETHEUS_URL",   "http://localhost:9090")
GRAFANA_URL      = os.getenv("GRAFANA_URL",      "http://localhost:3000")
GRAFANA_USER     = os.getenv("GRAFANA_USER",     "admin")
GRAFANA_PASS     = os.getenv("GRAFANA_PASS",     "mlops_pass")
LOKI_URL         = os.getenv("LOKI_URL",         "http://localhost:3100")
ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL", "http://localhost:9093")
PUSHGATEWAY_URL  = os.getenv("PUSHGATEWAY_URL",  "http://localhost:9091")

TIMEOUT = 5  # seconds per request

# ── Helpers ───────────────────────────────────────────────────

def prom_query(expr: str) -> list:
    """Execute a PromQL instant query. Returns result list."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": expr},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["data"]["result"]


def send_prediction(payload: dict = None) -> requests.Response:
    """Send a single prediction request."""
    if payload is None:
        payload = {
            "tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0,
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }
    return requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=TIMEOUT,
    )


# ════════════════════════════════════════════════════════════
# SERVICE HEALTH TESTS
# ════════════════════════════════════════════════════════════

class TestServiceHealth:

    def test_prometheus_healthy(self):
        """Prometheus responds to health check."""
        resp = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=TIMEOUT)
        assert resp.status_code == 200
        assert "Healthy" in resp.text

    def test_grafana_healthy(self):
        """Grafana API is reachable."""
        resp = requests.get(f"{GRAFANA_URL}/api/health", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("database") == "ok"

    def test_loki_ready(self):
        """Loki is ready to accept logs."""
        resp = requests.get(f"{LOKI_URL}/ready", timeout=TIMEOUT)
        assert resp.status_code == 200
        assert "ready" in resp.text.lower()

    def test_alertmanager_healthy(self):
        """Alertmanager is reachable."""
        resp = requests.get(f"{ALERTMANAGER_URL}/-/healthy", timeout=TIMEOUT)
        assert resp.status_code == 200

    def test_pushgateway_healthy(self):
        """Pushgateway is reachable."""
        resp = requests.get(f"{PUSHGATEWAY_URL}/-/healthy", timeout=TIMEOUT)
        assert resp.status_code == 200

    def test_api_healthy(self):
        """FastAPI /health returns healthy status."""
        resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "healthy"

    def test_api_metrics_endpoint_exists(self):
        """FastAPI /metrics endpoint returns Prometheus format."""
        resp = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_prometheus_targets_up(self):
        """All Prometheus scrape targets are in UP state."""
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/targets",
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        targets = resp.json()["data"]["activeTargets"]
        down_targets = [
            t["labels"].get("job", "unknown")
            for t in targets
            if t["health"] != "up"
        ]
        assert len(down_targets) == 0, f"Targets are DOWN: {down_targets}"


# ════════════════════════════════════════════════════════════
# GRAFANA TESTS
# ════════════════════════════════════════════════════════════

class TestGrafana:

    def test_prometheus_datasource_configured(self):
        """Prometheus datasource is provisioned in Grafana."""
        resp = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        datasource_types = [ds["type"] for ds in resp.json()]
        assert "prometheus" in datasource_types, "Prometheus datasource missing in Grafana"

    def test_loki_datasource_configured(self):
        """Loki datasource is provisioned in Grafana."""
        resp = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        datasource_types = [ds["type"] for ds in resp.json()]
        assert "loki" in datasource_types, "Loki datasource missing in Grafana"

    def test_ml_dashboard_exists(self):
        """ML Model Health dashboard is provisioned."""
        resp = requests.get(
            f"{GRAFANA_URL}/api/dashboards/uid/telco-ml-model-health",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, "ML Model Health dashboard not found"

    def test_api_dashboard_exists(self):
        """API Performance dashboard is provisioned."""
        resp = requests.get(
            f"{GRAFANA_URL}/api/dashboards/uid/telco-api-performance",
            auth=(GRAFANA_USER, GRAFANA_PASS),
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, "API Performance dashboard not found"


# ════════════════════════════════════════════════════════════
# PROMETHEUS RULES TESTS
# ════════════════════════════════════════════════════════════

class TestPrometheusRules:

    def test_alert_rules_loaded(self):
        """Alert rules are loaded in Prometheus."""
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/rules",
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        groups = resp.json()["data"]["groups"]
        rule_names = [
            rule["name"]
            for group in groups
            for rule in group["rules"]
        ]
        expected = [
            "PredictionLatencyHigh",
            "HighPredictionErrorRate",
            "FeatureDriftDetected",
            "ChurnAPIDown",
        ]
        for alert in expected:
            assert alert in rule_names, f"Alert rule '{alert}' not loaded"

    def test_recording_rules_loaded(self):
        """Recording rules are loaded and produce data."""
        # Give recording rules time to compute (interval: 1m)
        time.sleep(5)
        rules_resp = requests.get(f"{PROMETHEUS_URL}/api/v1/rules", timeout=TIMEOUT)
        groups = rules_resp.json()["data"]["groups"]
        recording_names = [
            rule["name"]
            for group in groups
            for rule in group["rules"]
            if rule["type"] == "recording"
        ]
        assert "job:predictions_total:rate5m" in recording_names


# ════════════════════════════════════════════════════════════
# METRICS FLOW TESTS
# ════════════════════════════════════════════════════════════

class TestMetricsFlow:

    def test_predictions_total_metric_exists(self):
        """predictions_total metric is exposed on /metrics."""
        resp = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        assert "predictions_total" in resp.text

    def test_prediction_latency_metric_exists(self):
        """prediction_latency_seconds histogram is exposed."""
        resp = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        assert "prediction_latency_seconds" in resp.text
        assert "prediction_latency_seconds_bucket" in resp.text

    def test_api_requests_total_metric_exists(self):
        """api_requests_total counter is exposed."""
        resp = requests.get(f"{API_URL}/metrics", timeout=TIMEOUT)
        assert "api_requests_total" in resp.text

    @pytest.mark.slow
    def test_prediction_increments_counter(self):
        """Sending a prediction increments predictions_total in Prometheus."""
        # Get baseline
        before = prom_query("sum(predictions_total)")
        before_val = float(before[0]["value"][1]) if before else 0.0

        # Send prediction
        resp = send_prediction()
        assert resp.status_code == 200

        # Wait for scrape
        time.sleep(20)

        # Check counter increased
        after = prom_query("sum(predictions_total)")
        after_val = float(after[0]["value"][1]) if after else 0.0

        assert after_val > before_val, (
            f"predictions_total did not increase: {before_val} → {after_val}"
        )

    @pytest.mark.slow
    def test_prediction_response_structure(self):
        """Prediction response has correct fields."""
        resp = send_prediction()
        assert resp.status_code == 200
        data = resp.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "outcome" in data
        assert "model_version" in data
        assert isinstance(data["churn_prediction"], bool)
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["outcome"] in ("churn", "no_churn")

    def test_invalid_prediction_returns_422(self):
        """Malformed request returns 422 Unprocessable Entity."""
        resp = requests.post(
            f"{API_URL}/predict",
            json={"tenure": "not-a-number"},
            timeout=TIMEOUT,
        )
        assert resp.status_code in (422, 500), (
            f"Expected 422, got {resp.status_code}"
        )

    def test_health_endpoint(self):
        """Health endpoint returns model version."""
        resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "model_version" in data
        assert data["status"] == "healthy"


# ════════════════════════════════════════════════════════════
# ALERTMANAGER TESTS
# ════════════════════════════════════════════════════════════

class TestAlertmanager:

    def test_alertmanager_api_reachable(self):
        """Alertmanager v2 API is reachable."""
        resp = requests.get(
            f"{ALERTMANAGER_URL}/api/v2/status",
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_can_send_test_alert(self):
        """Can POST a test alert to Alertmanager."""
        resp = requests.post(
            f"{ALERTMANAGER_URL}/api/v2/alerts",
            json=[{
                "labels": {
                    "alertname": "TestSmokeAlert",
                    "severity": "warning",
                    "job": "churn-api",
                },
                "annotations": {
                    "summary": "Automated smoke test alert",
                }
            }],
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200


# ════════════════════════════════════════════════════════════
# LOKI TESTS
# ════════════════════════════════════════════════════════════

class TestLoki:

    def test_loki_labels_exist(self):
        """Loki has received logs from churn-api job."""
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/label",
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_loki_query_endpoint(self):
        """Loki query endpoint responds successfully."""
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/query",
            params={"query": '{job="churn-api"}', "limit": 5},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200


# ════════════════════════════════════════════════════════════
# CLI RUNNER (when executed directly)
# ════════════════════════════════════════════════════════════

def run_smoke_tests():
    """Quick smoke test without pytest — prints pass/fail per test."""
    print("\n" + "═" * 60)
    print("MONITORING STACK SMOKE TESTS")
    print("═" * 60)

    tests = [
        ("Prometheus healthy",   lambda: requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=TIMEOUT).status_code == 200),
        ("Grafana healthy",      lambda: requests.get(f"{GRAFANA_URL}/api/health", timeout=TIMEOUT).json().get("database") == "ok"),
        ("Loki ready",           lambda: requests.get(f"{LOKI_URL}/ready", timeout=TIMEOUT).status_code == 200),
        ("Alertmanager healthy", lambda: requests.get(f"{ALERTMANAGER_URL}/-/healthy", timeout=TIMEOUT).status_code == 200),
        ("Pushgateway healthy",  lambda: requests.get(f"{PUSHGATEWAY_URL}/-/healthy", timeout=TIMEOUT).status_code == 200),
        ("API healthy",          lambda: requests.get(f"{API_URL}/health", timeout=TIMEOUT).json().get("status") == "healthy"),
        ("API /metrics works",   lambda: "predictions_total" in requests.get(f"{API_URL}/metrics", timeout=TIMEOUT).text),
        ("Prometheus scrapes API", lambda: bool(prom_query("up{job='churn-api'}"))),
        ("Alert rules loaded",   lambda: "PredictionLatencyHigh" in str(requests.get(f"{PROMETHEUS_URL}/api/v1/rules", timeout=TIMEOUT).json())),
        ("Grafana Prometheus DS", lambda: "prometheus" in str(requests.get(f"{GRAFANA_URL}/api/datasources", auth=(GRAFANA_USER, GRAFANA_PASS), timeout=TIMEOUT).json())),
        ("Grafana Loki DS",      lambda: "loki" in str(requests.get(f"{GRAFANA_URL}/api/datasources", auth=(GRAFANA_USER, GRAFANA_PASS), timeout=TIMEOUT).json())),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                print(f"  ✅  {name}")
                passed += 1
            else:
                print(f"  ❌  {name} — returned False")
                failed += 1
        except Exception as e:
            print(f"  ❌  {name} — {type(e).__name__}: {e}")
            failed += 1

    print("═" * 60)
    print(f"  Passed: {passed}/{len(tests)}")
    if failed:
        print(f"  Failed: {failed}/{len(tests)}")
    print("═" * 60 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
