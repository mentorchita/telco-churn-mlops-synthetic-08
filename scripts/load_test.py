"""
scripts/load_test.py
=====================
Async load generator for telco-churn prediction API.
Used in Demo 2 (generate traffic), Demo 4 (latency spike),
Demo 7 (inject errors), Demo 8 (A/B comparison).

Usage:
  # Basic: 100 requests
  python scripts/load_test.py

  # Custom count and URL
  python scripts/load_test.py --requests 500 --url http://localhost:8000

  # Inject bad requests for Demo 7
  python scripts/load_test.py --requests 100 --error-rate 0.2

  # Use slow endpoint for Demo 4
  python scripts/load_test.py --requests 200 --endpoint /predict-slow

Install:
  pip install aiohttp
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Install aiohttp: pip install aiohttp")
    raise

# ── Sample customer profiles ─────────────────────────────────
# Realistic telco customer data based on the synthetic dataset

SAMPLE_CUSTOMERS = [
    # High churn risk: month-to-month, electronic check, fiber optic
    {
        "tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    },
    # Low churn risk: two-year contract, bank transfer
    {
        "tenure": 48, "MonthlyCharges": 55.0, "TotalCharges": 2640.0,
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    },
    # Medium risk: one-year contract, credit card
    {
        "tenure": 18, "MonthlyCharges": 65.5, "TotalCharges": 1179.0,
        "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "Yes",
        "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
    },
    # No internet service
    {
        "tenure": 30, "MonthlyCharges": 25.0, "TotalCharges": 750.0,
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "No",
        "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service", "TechSupport": "No internet service",
        "StreamingTV": "No internet service", "StreamingMovies": "No internet service",
        "Contract": "Month-to-month", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
    },
]

# ── Bad/invalid payloads for error injection (Demo 7) ────────
BAD_PAYLOADS = [
    {"tenure": "not-a-number", "MonthlyCharges": 65.5},          # wrong type
    {"tenure": -999, "MonthlyCharges": 65.5, "Contract": "X"},   # negative tenure
    {"MonthlyCharges": 65.5},                                      # missing fields
    {},                                                            # empty body
    {"tenure": 999999, "MonthlyCharges": 999999},                  # out of range
]


@dataclass
class LoadTestStats:
    total:     int = 0
    success:   int = 0
    errors:    int = 0
    latencies: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record(self, status: int, latency_ms: float):
        self.total += 1
        if 200 <= status < 300:
            self.success += 1
        else:
            self.errors += 1
        self.latencies.append(latency_ms)

    def print_summary(self):
        elapsed = time.time() - self.start_time
        if not self.latencies:
            print("No requests completed.")
            return

        sorted_lat = sorted(self.latencies)
        p50  = sorted_lat[int(len(sorted_lat) * 0.50)]
        p95  = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99  = sorted_lat[int(len(sorted_lat) * 0.99)]
        avg  = sum(self.latencies) / len(self.latencies)
        rps  = self.total / elapsed if elapsed > 0 else 0

        print("\n" + "═" * 50)
        print("LOAD TEST RESULTS")
        print("═" * 50)
        print(f"  Total requests : {self.total}")
        print(f"  Success (2xx)  : {self.success} ({self.success/self.total*100:.1f}%)")
        print(f"  Errors         : {self.errors}  ({self.errors/self.total*100:.1f}%)")
        print(f"  Duration       : {elapsed:.1f}s")
        print(f"  Throughput     : {rps:.1f} req/s")
        print(f"\n  Latency (ms):")
        print(f"    p50  : {p50:.1f}")
        print(f"    p95  : {p95:.1f}")
        print(f"    p99  : {p99:.1f}")
        print(f"    avg  : {avg:.1f}")
        print(f"    min  : {min(self.latencies):.1f}")
        print(f"    max  : {max(self.latencies):.1f}")
        print("═" * 50)


async def send_request(
    session:      aiohttp.ClientSession,
    url:          str,
    payload:      dict,
    stats:        LoadTestStats,
    verbose:      bool = False,
):
    start = time.time()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            latency_ms = (time.time() - start) * 1000
            stats.record(resp.status, latency_ms)
            if verbose:
                body = await resp.json()
                print(f"  [{resp.status}] {latency_ms:.1f}ms → {body.get('outcome', '?')}")
    except asyncio.TimeoutError:
        stats.record(0, (time.time() - start) * 1000)
        if verbose:
            print(f"  [TIMEOUT] after {(time.time()-start)*1000:.0f}ms")
    except Exception as e:
        stats.record(0, (time.time() - start) * 1000)
        if verbose:
            print(f"  [ERROR] {e}")


async def run_load_test(
    base_url:    str,
    endpoint:    str,
    n_requests:  int,
    concurrency: int,
    error_rate:  float,
    verbose:     bool,
) -> LoadTestStats:
    stats = LoadTestStats()
    url = f"{base_url}{endpoint}"

    print(f"\n🚀 Load test starting")
    print(f"   Target:      {url}")
    print(f"   Requests:    {n_requests}")
    print(f"   Concurrency: {concurrency}")
    print(f"   Error rate:  {error_rate*100:.0f}%")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    progress_interval = max(1, n_requests // 10)

    async def bounded_request(i: int):
        async with semaphore:
            # Select payload
            if random.random() < error_rate:
                payload = random.choice(BAD_PAYLOADS)
            else:
                payload = random.choice(SAMPLE_CUSTOMERS).copy()
                # Add slight variation to tenure and charges
                payload["tenure"] = max(1, payload["tenure"] + random.randint(-3, 3))
                payload["MonthlyCharges"] = round(
                    payload["MonthlyCharges"] + random.uniform(-5, 5), 2
                )

            await send_request(session, url, payload, stats, verbose=verbose)

            if (i + 1) % progress_interval == 0:
                pct = (i + 1) / n_requests * 100
                elapsed = time.time() - stats.start_time
                rps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i+1}/{n_requests} ({pct:.0f}%) | {rps:.1f} req/s")

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_request(i) for i in range(n_requests)]
        await asyncio.gather(*tasks)

    stats.print_summary()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Load test for telco-churn predict API")
    parser.add_argument("--url",         default="http://localhost:8000", help="API base URL")
    parser.add_argument("--endpoint",    default="/predict",              help="Endpoint path")
    parser.add_argument("--requests",    type=int, default=100,           help="Total requests")
    parser.add_argument("--concurrency", type=int, default=10,            help="Concurrent requests")
    parser.add_argument("--error-rate",  type=float, default=0.0,         help="Fraction of bad requests (0-1)")
    parser.add_argument("--verbose",     action="store_true",             help="Print each response")
    args = parser.parse_args()

    asyncio.run(run_load_test(
        base_url=args.url,
        endpoint=args.endpoint,
        n_requests=args.requests,
        concurrency=args.concurrency,
        error_rate=args.error_rate,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
