#!/usr/bin/env bash
# =============================================================================
# scripts/demo_runner.sh
# One-stop script for all 8 classroom demos from Module 8 presentation.
#
# Usage:
#   chmod +x scripts/demo_runner.sh
#   ./scripts/demo_runner.sh <demo_number>
#
# Examples:
#   ./scripts/demo_runner.sh 1    # Deploy monitoring stack
#   ./scripts/demo_runner.sh 2    # Verify custom metrics
#   ./scripts/demo_runner.sh 4    # Inject latency spike
#   ./scripts/demo_runner.sh 5    # Run drift detection
#   ./scripts/demo_runner.sh 7    # Inject errors for log demo
#   ./scripts/demo_runner.sh all  # Run all demos sequentially
# =============================================================================

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASS="${GRAFANA_PASS:-mlops_pass}"
PUSHGATEWAY_URL="${PUSHGATEWAY_URL:-http://localhost:9091}"

# Colours
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[DEMO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }
sep()  { echo -e "${BOLD}═══════════════════════════════════════${NC}"; }

# ─────────────────────────────────────────────────────────────
demo_1() {
  sep
  echo -e "${BOLD}DEMO 1 — Deploy Full Monitoring Stack${NC}"
  sep

  log "Creating docker network (if needed)..."
  docker network create mlops-net 2>/dev/null && ok "Network created" || ok "Network already exists"

  log "Starting application services..."
  docker-compose up -d
  sleep 3

  log "Starting monitoring stack..."
  docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
  sleep 5

  log "Waiting for services to be healthy..."
  _wait_for "$PROMETHEUS_URL/-/healthy" "Prometheus" 30
  _wait_for "$GRAFANA_URL/api/health"   "Grafana"    30
  _wait_for "http://localhost:3100/ready" "Loki"      30

  sep
  log "Checking all Prometheus targets..."
  TARGETS=$(curl -s "$PROMETHEUS_URL/api/v1/targets" | python3 -c "
import sys, json
d = json.load(sys.stdin)['data']['activeTargets']
up   = [t['labels']['job'] for t in d if t['health']=='up']
down = [t['labels']['job'] for t in d if t['health']!='up']
print('UP:  ', ', '.join(sorted(set(up)))   if up   else 'none')
print('DOWN:', ', '.join(sorted(set(down))) if down else 'none')
  " 2>/dev/null || echo "Could not query targets")
  echo "$TARGETS"

  sep
  ok "✅ Stack deployed! Open:"
  echo "   Prometheus: $PROMETHEUS_URL"
  echo "   Grafana:    $GRAFANA_URL  (admin / $GRAFANA_PASS)"
  echo "   Loki:       http://localhost:3100"
  echo "   Alertmgr:   http://localhost:9093"
}

# ─────────────────────────────────────────────────────────────
demo_2() {
  sep
  echo -e "${BOLD}DEMO 2 — Send Traffic & Verify Custom Metrics${NC}"
  sep

  log "Checking /metrics endpoint is live..."
  if curl -sf "$API_URL/metrics" | grep -q "predictions_total"; then
    ok "predictions_total metric found!"
  else
    warn "predictions_total not found. Have you added src/metrics.py to app.py?"
    warn "See: src/app_with_metrics.py for integration guide."
  fi

  log "Sending 50 sample prediction requests..."
  _send_predictions 50

  log "Querying Prometheus for prediction rate..."
  sleep 3
  RATE=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode 'query=sum(rate(predictions_total[1m]))' \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['data']['result']
print(f'{float(r[0][\"value\"][1]):.3f} req/s' if r else 'no data yet — wait 15s')
  " 2>/dev/null)
  log "Current prediction rate: $RATE"

  log "Querying p99 latency..."
  P99=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode 'query=histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[1m]))' \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['data']['result']
print(f'{float(r[0][\"value\"][1])*1000:.1f}ms' if r else 'no data')
  " 2>/dev/null)
  log "p99 prediction latency: $P99"

  sep
  ok "✅ Open Prometheus: $PROMETHEUS_URL/graph"
  ok "   Try query: rate(predictions_total[1m])"
}

# ─────────────────────────────────────────────────────────────
demo_3() {
  sep
  echo -e "${BOLD}DEMO 3 — Import Grafana Dashboard via API${NC}"
  sep

  log "Checking if ml_model.json dashboard exists..."
  if [ -f "monitoring/grafana/dashboards/ml_model.json" ]; then
    log "Importing ML Model Health dashboard..."
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
      -X POST "$GRAFANA_URL/api/dashboards/import" \
      -H "Content-Type: application/json" \
      -u "$GRAFANA_USER:$GRAFANA_PASS" \
      -d @monitoring/grafana/dashboards/ml_model.json)
    [ "$STATUS" == "200" ] && ok "Dashboard imported (HTTP $STATUS)" || warn "Import returned HTTP $STATUS"
  else
    warn "monitoring/grafana/dashboards/ml_model.json not found"
    warn "Create a dashboard in Grafana UI and export it to that path"
  fi

  log "Checking datasources are provisioned..."
  DS=$(curl -s "$GRAFANA_URL/api/datasources" -u "$GRAFANA_USER:$GRAFANA_PASS" \
    | python3 -c "
import sys, json
ds = json.load(sys.stdin)
names = [d['name'] for d in ds]
print('Datasources:', ', '.join(names))
  " 2>/dev/null)
  ok "$DS"

  sep
  ok "✅ Open Grafana: $GRAFANA_URL"
  ok "   Login: $GRAFANA_USER / $GRAFANA_PASS"
  ok "   Navigate to Dashboards → MLOps folder"
}

# ─────────────────────────────────────────────────────────────
demo_4() {
  sep
  echo -e "${BOLD}DEMO 4 — Inject Latency Spike → Alert → Log Debug${NC}"
  sep

  log "PART 1: Sending traffic via /predict-slow endpoint..."
  log "(30% of requests will sleep 0.8-2.5s)"
  python3 scripts/load_test.py \
    --url "$API_URL" \
    --endpoint "/predict-slow" \
    --requests 100 \
    --concurrency 5

  sep
  log "PART 2: Watch alert status in Prometheus..."
  log "Opening: $PROMETHEUS_URL/alerts"
  echo ""
  log "Checking current p99 latency..."
  sleep 5
  P99=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode 'query=histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[2m]))' \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['data']['result']
if r:
    v = float(r[0]['value'][1])
    status = '🔴 ABOVE threshold' if v > 0.5 else '✅ Below threshold'
    print(f'{v*1000:.0f}ms  {status}')
else:
    print('No data — send more traffic first')
  " 2>/dev/null)
  echo "   p99 latency: $P99"

  sep
  log "PART 3: Investigate in Loki (open in browser)..."
  echo "   URL: $GRAFANA_URL/explore"
  echo "   Datasource: Loki"
  echo "   Query: {job=\"churn-api\"} | json | latency_ms > 500"

  sep
  ok "After demo: set INJECT_LATENCY=false and rebuild to resolve alert"
}

# ─────────────────────────────────────────────────────────────
demo_5() {
  sep
  echo -e "${BOLD}DEMO 5 — Detect Feature Drift${NC}"
  sep

  REFERENCE_PATH="${1:-data/reference/}"
  LIVE_PATH="${2:-data/live/}"

  log "PART 1: Generating reference data (2023)..."
  if [ ! -d "$REFERENCE_PATH" ] || [ -z "$(ls -A $REFERENCE_PATH 2>/dev/null)" ]; then
    python3 src/generate_dataset.py \
      --samples 10000 \
      --start-date 2023-01-01 \
      --end-date 2023-06-30 \
      --output-dir "$REFERENCE_PATH" && ok "Reference data generated"
  else
    ok "Reference data already exists in $REFERENCE_PATH"
  fi

  log "PART 2: Generating drifted live data (2024)..."
  if [ ! -d "$LIVE_PATH" ] || [ -z "$(ls -A $LIVE_PATH 2>/dev/null)" ]; then
    python3 src/generate_dataset.py \
      --samples 2000 \
      --start-date 2024-09-01 \
      --end-date 2024-12-31 \
      --output-dir "$LIVE_PATH" && ok "Live (drifted) data generated"
  else
    ok "Live data already exists in $LIVE_PATH"
  fi

  log "PART 3: Running drift detection..."
  python3 monitoring/drift_exporter.py \
    --reference "$REFERENCE_PATH" \
    --live "$LIVE_PATH" \
    --pushgateway "$PUSHGATEWAY_URL" \
    --model-version "v1.2"

  sep
  log "PART 4: Verify metrics in Prometheus..."
  sleep 5
  DRIFT=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode 'query=feature_drift_score' \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['data']['result']
if r:
    for m in sorted(r, key=lambda x: -float(x['value'][1])):
        feat = m['metric'].get('feature_name', '?')
        val  = float(m['value'][1])
        flag = '🔴' if val > 0.3 else '🟡' if val > 0.1 else '✅'
        print(f'  {flag} {feat:<25} KS={val:.4f}')
else:
    print('  No drift metrics found — check Pushgateway')
  " 2>/dev/null)
  echo "$DRIFT"

  sep
  ok "✅ Open Grafana Explore → Prometheus"
  ok "   Query: feature_drift_score"
  ok "   Create Bar Gauge panel with threshold 0.3 = red"
}

# ─────────────────────────────────────────────────────────────
demo_6() {
  sep
  echo -e "${BOLD}DEMO 6 — Fire Test Alert → Alertmanager Routing${NC}"
  sep

  log "Sending test alert to Alertmanager..."
  RESULT=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "http://localhost:9093/api/v2/alerts" \
    -H "Content-Type: application/json" \
    -d '[{
      "labels": {
        "alertname": "PredictionLatencyHigh",
        "severity":  "warning",
        "job":       "churn-api",
        "model_version": "v1.2"
      },
      "annotations": {
        "summary":     "Demo: p99 latency > 500ms",
        "description": "This is a manual test alert fired during DEMO 6"
      },
      "generatorURL": "http://prometheus:9090"
    }]')
  [ "$RESULT" == "200" ] && ok "Alert sent (HTTP $RESULT)" || warn "Unexpected HTTP $RESULT"

  log "Checking alert in Alertmanager..."
  sleep 2
  ALERTS=$(curl -s "http://localhost:9093/api/v2/alerts" \
    | python3 -c "
import sys, json
alerts = json.load(sys.stdin)
print(f'Active alerts: {len(alerts)}')
for a in alerts:
    name = a.get('labels', {}).get('alertname', '?')
    sev  = a.get('labels', {}).get('severity', '?')
    print(f'  → {name} [{sev}]')
  " 2>/dev/null)
  echo "$ALERTS"

  sep
  ok "Open Alertmanager UI: http://localhost:9093/#/alerts"
  ok "Open Prometheus Alerts: $PROMETHEUS_URL/alerts"

  log "Silencing the test alert..."
  curl -s -X POST "http://localhost:9093/api/v2/silences" \
    -H "Content-Type: application/json" \
    -d '{
      "matchers": [{"name": "alertname", "value": "PredictionLatencyHigh", "isRegex": false}],
      "startsAt":  "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'",
      "endsAt":    "'"$(date -u -d '+1 hour' +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -v+1H +%Y-%m-%dT%H:%M:%SZ)"'",
      "createdBy": "demo-script",
      "comment":   "Demo 6 test silence"
    }' | python3 -c "import sys,json; d=json.load(sys.stdin); print('Silence ID:', d.get('silenceID','?'))" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────
demo_7() {
  sep
  echo -e "${BOLD}DEMO 7 — Inject Errors → Debug via Loki LogQL${NC}"
  sep

  log "PART 1: Sending 50 valid requests (baseline)..."
  _send_predictions 50 >/dev/null 2>&1
  ok "Baseline traffic sent"

  log "PART 2: Sending 20 malformed requests (tenure as string)..."
  for i in $(seq 1 20); do
    curl -s -X POST "$API_URL/predict" \
      -H "Content-Type: application/json" \
      -d '{"tenure": "not-a-number", "MonthlyCharges": 65.5, "Contract": "Month-to-month"}' \
      >/dev/null 2>&1
  done
  ok "20 bad requests sent (tenure: string)"

  log "PART 3: Sending 10 requests with negative tenure..."
  for i in $(seq 1 10); do
    curl -s -X POST "$API_URL/predict" \
      -H "Content-Type: application/json" \
      -d '{"tenure": -999, "MonthlyCharges": 65.5, "Contract": "Month-to-month"}' \
      >/dev/null 2>&1
  done
  ok "10 bad requests sent (tenure: -999)"

  log "PART 4: Checking error rate in Prometheus..."
  sleep 5
  ERATE=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode 'query=sum(rate(api_requests_total{status_code=~"4.."}[2m])) / sum(rate(api_requests_total[2m])) * 100' \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['data']['result']
if r:
    v = float(r[0]['value'][1])
    print(f'{v:.1f}%  {\"🔴 High\" if v > 10 else \"🟡 Moderate\"}')
else:
    print('No data')
  " 2>/dev/null)
  log "4xx error rate: $ERATE"

  sep
  ok "✅ Now investigate in Grafana Explore → Loki:"
  echo ""
  echo "   Step 1 - Find all errors:"
  echo '   {job="churn-api"} |= "ERROR"'
  echo ""
  echo "   Step 2 - Parse JSON and filter 4xx:"
  echo '   {job="churn-api"} | json | status_code >= 400'
  echo ""
  echo "   Step 3 - See error message:"
  echo '   {job="churn-api"} | json | status_code = "422" | line_format "{{.message}}"'
}

# ─────────────────────────────────────────────────────────────
demo_8() {
  sep
  echo -e "${BOLD}DEMO 8 — A/B Test: Compare Two Model Versions${NC}"
  sep
  warn "This demo requires two model versions registered in MLflow."
  warn "See deployment/ab_router.py for the router implementation."
  echo ""

  log "Sending 300 requests (A/B split should produce v1.2 and v1.3 metrics)..."
  python3 scripts/load_test.py \
    --url "$API_URL" \
    --requests 300 \
    --concurrency 10

  sep
  log "Comparing model versions in Prometheus..."
  sleep 5
  VERSIONS=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode 'query=sum by(model_version)(rate(predictions_total[2m]))' \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d['data']['result']
if r:
    for m in sorted(r, key=lambda x: x['metric'].get('model_version','')):
        ver  = m['metric'].get('model_version', '?')
        rate = float(m['value'][1])
        print(f'  {ver}: {rate:.3f} req/s')
else:
    print('  Only one model version found — configure ABRouter first')
  " 2>/dev/null)
  echo "$VERSIONS"

  sep
  ok "✅ Grafana A/B comparison:"
  ok "   Add Variable: model_version → label_values(predictions_total, model_version)"
  ok "   Panel 1: rate(predictions_total[5m]) by model_version"
  ok "   Panel 2: histogram_quantile(0.99, ...) by model_version"
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

_wait_for() {
  local url="$1" name="$2" max="${3:-30}"
  local i=0
  while [ $i -lt $max ]; do
    if curl -sf "$url" >/dev/null 2>&1; then
      ok "$name is healthy"
      return 0
    fi
    echo -n "."
    sleep 1
    i=$((i+1))
  done
  warn "$name did not respond after ${max}s"
  return 1
}

_send_predictions() {
  local n="${1:-20}"
  local CONTRACTS=("Month-to-month" "One year" "Two year")
  local INTERNET=("Fiber optic" "DSL" "No")
  local PAYMENT=("Electronic check" "Mailed check" "Bank transfer (automatic)" "Credit card (automatic)")

  for i in $(seq 1 "$n"); do
    TENURE=$((RANDOM % 72 + 1))
    CHARGES=$(python3 -c "import random; print(round(random.uniform(18, 120), 2))")
    TOTAL=$(python3 -c "print(round($TENURE * $CHARGES, 2))")
    CONTRACT=${CONTRACTS[$((RANDOM % 3))]}
    INET=${INTERNET[$((RANDOM % 3))]}
    PAY=${PAYMENT[$((RANDOM % 4))]}

    curl -s -X POST "$API_URL/predict" \
      -H "Content-Type: application/json" \
      -d "{
        \"tenure\": $TENURE,
        \"MonthlyCharges\": $CHARGES,
        \"TotalCharges\": $TOTAL,
        \"gender\": \"Male\",
        \"SeniorCitizen\": 0,
        \"Partner\": \"Yes\",
        \"Dependents\": \"No\",
        \"PhoneService\": \"Yes\",
        \"MultipleLines\": \"No\",
        \"InternetService\": \"$INET\",
        \"OnlineSecurity\": \"No\",
        \"OnlineBackup\": \"No\",
        \"DeviceProtection\": \"No\",
        \"TechSupport\": \"No\",
        \"StreamingTV\": \"No\",
        \"StreamingMovies\": \"No\",
        \"Contract\": \"$CONTRACT\",
        \"PaperlessBilling\": \"Yes\",
        \"PaymentMethod\": \"$PAY\"
      }" >/dev/null 2>&1 &
  done
  wait
  ok "Sent $n prediction requests"
}

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

case "${1:-help}" in
  1)   demo_1 ;;
  2)   demo_2 ;;
  3)   demo_3 ;;
  4)   demo_4 ;;
  5)   demo_5 "${2:-}" "${3:-}" ;;
  6)   demo_6 ;;
  7)   demo_7 ;;
  8)   demo_8 ;;
  all)
    demo_1; sleep 10
    demo_2; sleep 5
    demo_5; sleep 5
    demo_7
    ;;
  help|*)
    echo ""
    echo -e "${BOLD}Module 8 Demo Runner — telco-churn-mlops${NC}"
    echo ""
    echo "Usage: $0 <demo_number>"
    echo ""
    echo "  1  Deploy full monitoring stack (Docker Compose)"
    echo "  2  Send traffic & verify custom metrics in Prometheus"
    echo "  3  Import Grafana dashboard via API"
    echo "  4  Inject latency spike → trigger alert → debug via logs"
    echo "  5  Run drift detection & push to Prometheus"
    echo "  6  Fire test alert → Alertmanager routing"
    echo "  7  Inject HTTP errors → debug via Loki LogQL"
    echo "  8  A/B model comparison (requires two MLflow versions)"
    echo "  all  Run demos 1, 2, 5, 7 in sequence"
    echo ""
    echo "Environment variables:"
    echo "  API_URL          (default: http://localhost:8000)"
    echo "  PROMETHEUS_URL   (default: http://localhost:9090)"
    echo "  GRAFANA_URL      (default: http://localhost:3000)"
    echo "  PUSHGATEWAY_URL  (default: http://localhost:9091)"
    echo ""
    ;;
esac
