"""
monitoring/drift_exporter.py
=============================
Computes feature drift between reference (training) data and live data.
Pushes results to Prometheus Pushgateway.

Run as:
  - Airflow DAG (scheduled daily)
  - Kubernetes CronJob
  - Manual: python monitoring/drift_exporter.py

Install:
  pip install scipy prometheus-client pandas mlflow

Demo 5 usage:
  # Generate reference data (2023)
  python src/generate_dataset.py --start-date 2023-01-01 --end-date 2023-06-30 --output-dir data/reference/

  # Generate drifted live data (2024)
  python src/generate_dataset.py --start-date 2024-09-01 --end-date 2024-12-31 --output-dir data/live/

  # Run drift detection
  python monitoring/drift_exporter.py \\
      --reference data/reference/ \\
      --live      data/live/ \\
      --pushgateway http://localhost:9091
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    push_to_gateway,
)

# Optional MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger("drift-exporter")

# ── Default config ───────────────────────────────────────────
PUSHGATEWAY_URL     = os.getenv("PUSHGATEWAY_URL",     "http://localhost:9091")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_VERSION       = os.getenv("MODEL_VERSION",       "v1.2")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/reference/")
LIVE_DATA_PATH      = os.getenv("LIVE_DATA_PATH",      "data/live/")

# ── Feature definitions ───────────────────────────────────────
# Numerical features → KS test
NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Categorical features → PSI (Population Stability Index)
CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "gender",
    "Partner",
    "Dependents",
]

# ═══════════════════════════════════════════════════════════════
# DRIFT DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def compute_ks_drift(reference: pd.Series, live: pd.Series) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov two-sample test for numerical feature drift.

    Returns:
        ks_statistic: 0 = identical distributions, 1 = completely different
        p_value:      probability of observing this KS stat under H0 (no drift)

    Interpretation:
        ks_stat < 0.1:  no drift
        ks_stat < 0.2:  minor drift (monitor)
        ks_stat < 0.3:  moderate drift (warning)
        ks_stat >= 0.3: significant drift (alert)
        ks_stat >= 0.5: severe drift (retrain immediately)
    """
    ref_clean  = reference.dropna()
    live_clean = live.dropna()

    if len(ref_clean) == 0 or len(live_clean) == 0:
        logger.warning(f"Empty series for KS test")
        return 0.0, 1.0

    ks_stat, p_value = stats.ks_2samp(ref_clean, live_clean)
    return float(ks_stat), float(p_value)


def compute_psi(reference: pd.Series, live: pd.Series) -> float:
    """
    Population Stability Index for categorical feature drift.

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Interpretation:
        PSI < 0.10:  negligible change (stable)
        PSI < 0.20:  moderate change (monitor)
        PSI < 0.25:  significant change (warning)
        PSI >= 0.25: major shift (retrain recommended)
    """
    EPSILON = 1e-6  # avoid log(0)

    ref_dist  = reference.value_counts(normalize=True)
    live_dist = live.value_counts(normalize=True)

    # Align categories (handle new categories in live data)
    all_cats = set(ref_dist.index) | set(live_dist.index)
    ref_dist  = ref_dist.reindex(all_cats, fill_value=EPSILON)
    live_dist = live_dist.reindex(all_cats, fill_value=EPSILON)

    psi = ((live_dist - ref_dist) * np.log(live_dist / ref_dist)).sum()
    return float(abs(psi))


def compute_data_quality(series: pd.Series, feature_name: str) -> dict:
    """
    Compute data quality metrics for a feature.

    Returns dict with null_rate, out_of_range_rate, quality_score.
    """
    total = len(series)
    null_count = series.isnull().sum()
    null_rate = null_count / total if total > 0 else 0.0

    quality_score = 1.0 - null_rate  # simple quality: fraction of non-null

    return {
        "null_count":    int(null_count),
        "null_rate":     float(null_rate),
        "quality_score": float(quality_score),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN DRIFT CHECK FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_drift_check(
    reference_path: str,
    live_path: str,
    pushgateway_url: str,
    model_version: str,
):
    """
    Full drift check pipeline:
    1. Load reference and live data
    2. Compute KS drift for numerical features
    3. Compute PSI drift for categorical features
    4. Push metrics to Prometheus Pushgateway
    5. Log results to MLflow (optional)

    Returns:
        dict of drift results
    """
    logger.info(f"Starting drift check | model={model_version}")
    start_time = time.time()

    # ── Load data ────────────────────────────────────────────
    logger.info(f"Loading reference data from: {reference_path}")
    ref_df = _load_dataframe(reference_path)
    logger.info(f"Loading live data from: {live_path}")
    live_df = _load_dataframe(live_path)

    if ref_df is None or live_df is None:
        logger.error("Failed to load data — aborting drift check")
        return {}

    logger.info(f"Reference rows: {len(ref_df):,} | Live rows: {len(live_df):,}")

    # ── Set up Prometheus registry ───────────────────────────
    registry = CollectorRegistry()

    drift_gauge = Gauge(
        "feature_drift_score",
        "Kolmogorov-Smirnov statistic for numerical feature drift",
        ["feature_name", "model_version"],
        registry=registry,
    )
    psi_gauge = Gauge(
        "feature_psi_score",
        "PSI score for categorical feature drift",
        ["feature_name", "model_version"],
        registry=registry,
    )
    quality_gauge = Gauge(
        "data_quality_score",
        "Fraction of valid (non-null) values for this feature",
        ["feature_name"],
        registry=registry,
    )
    drift_check_counter = Counter(
        "drift_checks_total",
        "Total drift checks performed",
        ["status"],
        registry=registry,
    )
    check_duration_gauge = Gauge(
        "drift_check_duration_seconds",
        "Time to complete drift check",
        registry=registry,
    )

    results = {}

    # ── Numerical features: KS test ──────────────────────────
    logger.info("Computing KS drift for numerical features...")
    for feature in NUMERICAL_FEATURES:
        if feature not in ref_df.columns or feature not in live_df.columns:
            logger.warning(f"Feature '{feature}' not found in data — skipping")
            continue

        ks_stat, p_value = compute_ks_drift(ref_df[feature], live_df[feature])

        drift_gauge.labels(
            feature_name=feature,
            model_version=model_version,
        ).set(ks_stat)

        # Data quality
        quality = compute_data_quality(live_df[feature], feature)
        quality_gauge.labels(feature_name=feature).set(quality["quality_score"])

        results[feature] = {
            "type":        "numerical",
            "ks_stat":     ks_stat,
            "p_value":     p_value,
            "drift_level": _classify_ks(ks_stat),
        }

        level = _classify_ks(ks_stat)
        emoji = {"none": "✅", "minor": "🟡", "moderate": "🟠", "severe": "🔴"}.get(level, "")
        logger.info(
            f"  {emoji} {feature}: KS={ks_stat:.4f} (p={p_value:.4f}) → {level}"
        )

    # ── Categorical features: PSI ────────────────────────────
    logger.info("Computing PSI drift for categorical features...")
    for feature in CATEGORICAL_FEATURES:
        if feature not in ref_df.columns or feature not in live_df.columns:
            logger.warning(f"Feature '{feature}' not found — skipping")
            continue

        psi = compute_psi(ref_df[feature], live_df[feature])

        psi_gauge.labels(
            feature_name=feature,
            model_version=model_version,
        ).set(psi)

        # Data quality
        quality = compute_data_quality(live_df[feature], feature)
        quality_gauge.labels(feature_name=feature).set(quality["quality_score"])

        results[feature] = {
            "type":        "categorical",
            "psi":         psi,
            "drift_level": _classify_psi(psi),
        }

        level = _classify_psi(psi)
        emoji = {"stable": "✅", "monitor": "🟡", "warning": "🟠", "critical": "🔴"}.get(level, "")
        logger.info(
            f"  {emoji} {feature}: PSI={psi:.4f} → {level}"
        )

    # ── Duration metric ──────────────────────────────────────
    duration = time.time() - start_time
    check_duration_gauge.set(duration)
    drift_check_counter.labels(status="success").inc()

    # ── Push to Pushgateway ──────────────────────────────────
    logger.info(f"Pushing metrics to Pushgateway: {pushgateway_url}")
    try:
        push_to_gateway(
            pushgateway_url,
            job="drift-checker",
            registry=registry,
        )
        logger.info("✅ Metrics pushed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to push to Pushgateway: {e}")
        logger.info("Continuing — metrics computed but not pushed")

    # ── Log to MLflow (optional) ─────────────────────────────
    if MLFLOW_AVAILABLE:
        try:
            _log_to_mlflow(results, model_version, duration)
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-critical): {e}")

    # ── Summary ──────────────────────────────────────────────
    _print_summary(results, duration)

    return results


def _load_dataframe(path: str) -> pd.DataFrame | None:
    """Load CSV from a directory or file path."""
    p = Path(path)
    if p.is_file():
        return pd.read_csv(p)
    elif p.is_dir():
        csv_files = list(p.glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {path}")
            return None
        # Load the most recently modified CSV
        latest = max(csv_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Loading: {latest}")
        return pd.read_csv(latest)
    else:
        logger.error(f"Path not found: {path}")
        return None


def _classify_ks(ks_stat: float) -> str:
    if ks_stat < 0.10:  return "none"
    if ks_stat < 0.20:  return "minor"
    if ks_stat < 0.30:  return "moderate"
    return "severe"


def _classify_psi(psi: float) -> str:
    if psi < 0.10:  return "stable"
    if psi < 0.20:  return "monitor"
    if psi < 0.25:  return "warning"
    return "critical"


def _log_to_mlflow(results: dict, model_version: str, duration: float):
    """Log drift metrics to MLflow for audit trail."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("drift-monitoring")

    with mlflow.start_run(run_name=f"drift_check_{model_version}"):
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("check_type", "scheduled")

        metrics_to_log = {"check_duration_seconds": duration}
        for feature, data in results.items():
            if data["type"] == "numerical":
                metrics_to_log[f"ks_{feature}"] = data["ks_stat"]
            else:
                metrics_to_log[f"psi_{feature}"] = data["psi"]

        mlflow.log_metrics(metrics_to_log)
        logger.info("MLflow run logged successfully")


def _print_summary(results: dict, duration: float):
    """Print a formatted drift summary."""
    print("\n" + "═" * 60)
    print("DRIFT CHECK SUMMARY")
    print("═" * 60)

    alerts = []
    for feature, data in results.items():
        level = data.get("drift_level", "unknown")
        if data["type"] == "numerical":
            score = f"KS={data['ks_stat']:.4f}"
        else:
            score = f"PSI={data['psi']:.4f}"

        indicator = {"none": "✅", "minor": "🟡", "moderate": "🟠", "severe": "🔴",
                     "stable": "✅", "monitor": "🟡", "warning": "🟠", "critical": "🔴"}.get(level, "❓")

        print(f"  {indicator}  {feature:<25} {score:<15} [{level.upper()}]")

        if level in ("severe", "critical", "warning", "moderate"):
            alerts.append(f"{feature} ({score})")

    print(f"\n  ⏱  Duration: {duration:.2f}s")
    if alerts:
        print(f"\n  ⚠️  Features needing attention: {', '.join(alerts)}")
    else:
        print(f"\n  ✅  All features stable")
    print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute feature drift and push to Prometheus Pushgateway"
    )
    parser.add_argument(
        "--reference",
        default=REFERENCE_DATA_PATH,
        help="Path to reference (training) data directory or CSV",
    )
    parser.add_argument(
        "--live",
        default=LIVE_DATA_PATH,
        help="Path to live (production) data directory or CSV",
    )
    parser.add_argument(
        "--pushgateway",
        default=PUSHGATEWAY_URL,
        help="Prometheus Pushgateway URL",
    )
    parser.add_argument(
        "--model-version",
        default=MODEL_VERSION,
        help="Model version label for metrics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute drift but do not push to Pushgateway",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN mode — will not push to Pushgateway")

    results = run_drift_check(
        reference_path=args.reference,
        live_path=args.live,
        pushgateway_url=args.pushgateway if not args.dry_run else "http://dry-run",
        model_version=args.model_version,
    )

    # Exit with non-zero code if critical drift detected
    critical = any(
        d.get("drift_level") in ("severe", "critical")
        for d in results.values()
    )
    if critical:
        logger.warning("Critical drift detected! Consider retraining.")
        sys.exit(2)    # exit code 2 = drift warning (use in CI/CD checks)


if __name__ == "__main__":
    main()
