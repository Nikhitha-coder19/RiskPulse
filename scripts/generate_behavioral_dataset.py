#!/usr/bin/env python3
"""Generate a synthetic behavioral risk dataset for RiskPulse.

The raw archive dataset is used only as a statistical inspiration:
- payment methods are balanced across PayPal, bank transfer, credit card, and debit card
- product categories are balanced across five categories
- device share is roughly one-third each for desktop, mobile, and tablet
- quantity is concentrated in 1..5 with a median around 3
- transaction amount is right-skewed, with a median near 150 and a long tail
- customer age and account age roughly match the original dataset's ranges
- transaction time is spread across the day and across a multi-month timeline

This generator intentionally creates a synthetic behavioral world with persistent customers,
merchants, devices, and IPs. History features are computed strictly from earlier transactions,
never from future rows, and never from the target label. The fraud rate is tuned to stay in the
realistic 5-8% range while still preserving multiple fraud mechanisms: individual fraud,
coordinated fraud, suspicious merchants, and legitimate high-value purchases.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SEED = 42
TARGET_ROWS = 300_000
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Data" / "behavioral"
CSV_PATH = OUTPUT_DIR / "behavioral_transactions.csv"
SUMMARY_PATH = OUTPUT_DIR / "generation_summary.json"


def sample_timestamp(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    return datetime.fromtimestamp(rng.uniform(start_ts, end_ts))


def sample_amount(rng: np.random.Generator, base_amount: float, high_value: bool = False, suspicious: bool = False) -> float:
    if high_value:
        amt = float(np.clip(rng.lognormal(mean=6.2, sigma=0.45), 700, 8000))
        return round(float(amt), 2)
    if suspicious:
        amt = float(np.clip(rng.lognormal(mean=5.8, sigma=0.7), 200, 3000))
        return round(float(amt), 2)
    # Calibrated to the original dataset shape: skewed, median ~150, 95% below ~650, long tail
    amt = float(np.clip(rng.lognormal(mean=4.9, sigma=0.8), 10, 11000))
    amt *= max(0.6, min(1.8, base_amount / 160.0))
    return round(float(amt), 2)


def sample_quantity(rng: np.random.Generator) -> int:
    weights = np.array([0.15, 0.25, 0.30, 0.20, 0.10])
    choices = np.array([1, 2, 3, 4, 5])
    return int(rng.choice(choices, p=weights))


def sample_payment_method(rng: np.random.Generator) -> str:
    return rng.choice(["PayPal", "bank transfer", "credit card", "debit card"], p=np.array([0.25, 0.25, 0.25, 0.25]))


def sample_product_category(rng: np.random.Generator) -> str:
    return rng.choice(["clothing", "electronics", "health & beauty", "home & garden", "toys & games"], p=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))


def sample_device(rng: np.random.Generator, *, preferred_family: str | None = None) -> str:
    families = np.array(["desktop", "mobile", "tablet"])
    family = preferred_family if preferred_family in families else rng.choice(families, p=np.array([0.333, 0.334, 0.333]))
    device_id = int(rng.integers(1, 25000))
    return f"{family}_device_{device_id:06d}"


def make_customer_profiles(rng: np.random.Generator, n_customers: int) -> Dict[str, Dict[str, object]]:
    profiles: Dict[str, Dict[str, object]] = {}
    for idx in range(n_customers):
        cid = f"cust_{idx:06d}"
        profile = {
            "customer_id": cid,
            "customer_age": int(np.clip(rng.normal(34.5, 10.0), 18, 75)),
            "account_age_days": int(np.clip(rng.normal(180, 110), 1, 365)),
            "location": rng.choice([
                "New Michael", "South Michael", "East Michael", "Port Michael", "West Michael",
                "North Michael", "Lake Michael", "Michaelmouth", "West David", "New David",
                "Ramosfort", "Brockburgh", "Carneyfurt", "Amandaborough", "Port Emily",
                "Lynnberg", "Herreramouth", "South Nicole", "Davismouth", "East Timothy",
            ]),
            "preferred_device": sample_device(rng),
            "preferred_ip": f"ip_{rng.integers(1, 12000):06d}",
            "preferred_shipping_address": f"ship_{rng.integers(1, 50000):08d}",
            "preferred_billing_address": f"bill_{rng.integers(1, 50000):08d}",
            "base_amount": float(np.clip(rng.normal(180, 120), 30, 600)),
            "volatility": float(np.clip(rng.uniform(0.15, 0.75), 0.1, 1.0)),
            "activity_rate": float(np.clip(rng.uniform(0.3, 2.5), 0.1, 5.0)),
            "first_seen": datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 700))),
            "preferred_merchants": [],
        }
        profiles[cid] = profile
    return profiles


def make_merchant_profiles(rng: np.random.Generator, n_merchants: int, suspicious_count: int) -> Dict[str, Dict[str, object]]:
    profiles: Dict[str, Dict[str, object]] = {}
    for idx in range(n_merchants):
        mid = f"merchant_{idx:06d}"
        suspicious = idx < suspicious_count
        profiles[mid] = {
            "merchant_id": mid,
            "suspicious": suspicious,
            "baseline_volume": float(np.clip(rng.lognormal(mean=4.6, sigma=0.8), 40, 4000)),
            "start_time": datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 500))),
            "risk_shift": float(rng.uniform(0.9, 1.6)) if suspicious else 1.0,
        }
    return profiles


def sample_authentication_status(rng: np.random.Generator, *, suspicious: bool = False, new_customer: bool = False) -> str:
    if suspicious:
        return rng.choice(["failed", "authenticated", "not_required"], p=np.array([0.60, 0.30, 0.10]))
    if new_customer:
        return rng.choice(["authenticated", "failed", "not_required"], p=np.array([0.65, 0.20, 0.15]))
    return rng.choice(["authenticated", "not_required", "failed"], p=np.array([0.78, 0.18, 0.04]))


def generate_event(
    *,
    customer_id: str,
    merchant_id: str,
    timestamp: datetime,
    transaction_amount: float,
    quantity: int,
    payment_method: str,
    product_category: str,
    customer_age: int,
    account_age_days: int,
    location: str,
    device_id: str,
    ip_id: str,
    shipping_address_id: str,
    billing_address_id: str,
    authentication_status: str,
    is_fraud: int,
    scenario_type: str,
) -> Dict[str, object]:
    return {
        "transaction_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "timestamp": timestamp,
        "transaction_amount": round(float(transaction_amount), 2),
        "quantity": int(quantity),
        "payment_method": payment_method,
        "product_category": product_category,
        "customer_age": int(customer_age),
        "account_age_days": int(account_age_days),
        "customer_location": location,
        "device_id": device_id,
        "ip_id": ip_id,
        "shipping_address_id": shipping_address_id,
        "billing_address_id": billing_address_id,
        "authentication_status": authentication_status,
        "is_fraud": int(is_fraud),
        "scenario_type": scenario_type,
    }


def generate_normal_events(rng: np.random.Generator, customer_profiles: Dict[str, Dict[str, object]], merchant_ids: List[str], target_count: int) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    customer_ids = np.array(list(customer_profiles.keys()))
    for _ in range(target_count):
        customer_id = rng.choice(customer_ids)
        profile = customer_profiles[customer_id]
        merchant_id = rng.choice(merchant_ids)
        ts = sample_timestamp(rng, profile["first_seen"], datetime(2024, 12, 31))
        quantity = sample_quantity(rng)
        base = float(profile["base_amount"]) * (0.8 + 0.6 * profile["volatility"])
        amount = float(np.clip(rng.normal(base, max(20.0, base * 0.55)), 15.0, 1500.0))
        preferred_family = str(profile["preferred_device"]).split("_")[0]
        device_id = profile["preferred_device"] if rng.random() < 0.85 else sample_device(rng, preferred_family=preferred_family)
        ip_id = profile["preferred_ip"] if rng.random() < 0.65 else f"ip_{rng.integers(1, 12000):06d}"
        shipping = (
            profile["preferred_shipping_address"]
            if rng.random() < 0.90
            else f"ship_{rng.integers(1, 50000):08d}"
        )

        billing = (
            profile["preferred_billing_address"]
            if rng.random() < 0.90
            else f"bill_{rng.integers(1, 50000):08d}"
        )
        auth = sample_authentication_status(rng)
        events.append(
            generate_event(
                customer_id=customer_id,
                merchant_id=merchant_id,
                timestamp=ts,
                transaction_amount=amount,
                quantity=quantity,
                payment_method=sample_payment_method(rng),
                product_category=sample_product_category(rng),
                customer_age=int(profile["customer_age"]),
                account_age_days=int(profile["account_age_days"]),
                location=str(profile["location"]),
                device_id=device_id,
                ip_id=ip_id,
                shipping_address_id=shipping,
                billing_address_id=billing,
                authentication_status=auth,
                is_fraud=0,
                scenario_type="normal",
            )
        )
    return events


def generate_legitimate_high_value_events(rng: np.random.Generator, customer_profiles: Dict[str, Dict[str, object]], merchant_ids: List[str], target_count: int) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for _ in range(target_count):
        customer_id = rng.choice(list(customer_profiles.keys()))
        profile = customer_profiles[customer_id]
        merchant_id = rng.choice(merchant_ids)
        ts = sample_timestamp(rng, datetime(2023, 3, 1), datetime(2024, 12, 31))
        quantity = int(rng.choice([1, 2, 3, 4, 5], p=np.array([0.08, 0.15, 0.27, 0.25, 0.25])))
        amount = float(np.clip(rng.lognormal(mean=6.2, sigma=0.5), 700, 8000)) * (0.8 + 0.12 * quantity)
        preferred_family = str(profile["preferred_device"]).split("_")[0]
        device_id = profile["preferred_device"] if rng.random() < 0.9 else sample_device(rng, preferred_family=preferred_family)
        ip_id = profile["preferred_ip"] if rng.random() < 0.7 else f"ip_{rng.integers(1, 12000):06d}"
        shipping = (
            profile["preferred_shipping_address"]
            if rng.random() < 0.92
            else f"ship_{rng.integers(1, 50000):08d}"
        )

        billing = (
            profile["preferred_billing_address"]
            if rng.random() < 0.92
            else f"bill_{rng.integers(1, 50000):08d}"
        )
        auth = sample_authentication_status(rng)
        events.append(
            generate_event(
                customer_id=customer_id,
                merchant_id=merchant_id,
                timestamp=ts,
                transaction_amount=amount,
                quantity=quantity,
                payment_method=sample_payment_method(rng),
                product_category=sample_product_category(rng),
                customer_age=int(profile["customer_age"]),
                account_age_days=int(profile["account_age_days"]),
                location=str(profile["location"]),
                device_id=device_id,
                ip_id=ip_id,
                shipping_address_id=shipping,
                billing_address_id=billing,
                authentication_status=auth,
                is_fraud=0,
                scenario_type="legitimate_high_value",
            )
        )
    return events


def generate_new_customer_events(rng: np.random.Generator, customer_profiles: Dict[str, Dict[str, object]], merchant_ids: List[str], target_count: int) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for idx in range(target_count):
        new_customer_id = f"cust_{len(customer_profiles)+idx:06d}"
        profile = {
            "customer_id": new_customer_id,
            "customer_age": int(np.clip(rng.normal(34.5, 10.0), 18, 75)),
            "account_age_days": 1,
            "location": rng.choice([
                "New Michael", "South Michael", "East Michael", "Port Michael", "West Michael",
                "North Michael", "Lake Michael", "Michaelmouth", "West David", "New David",
                "Ramosfort", "Brockburgh", "Carneyfurt", "Amandaborough", "Port Emily",
                "Lynnberg", "Herreramouth", "South Nicole", "Davismouth", "East Timothy",
            ]),
            "preferred_device": sample_device(rng),
            "preferred_ip": f"ip_{rng.integers(1, 12000):06d}",
            "preferred_shipping_address": f"ship_{rng.integers(1, 50000):08d}",
            "preferred_billing_address": f"bill_{rng.integers(1, 50000):08d}",
            "base_amount": float(np.clip(rng.normal(130, 90), 20, 500)),
            "volatility": float(np.clip(rng.uniform(0.2, 0.9), 0.1, 1.0)),
            "first_seen": datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 700))),
        }
        customer_profiles[new_customer_id] = profile
        merchant_id = rng.choice(merchant_ids)
        ts = sample_timestamp(rng, profile["first_seen"], datetime(2024, 12, 31))
        quantity = sample_quantity(rng)
        amount = float(np.clip(rng.normal(profile["base_amount"], max(30.0, profile["base_amount"] * 0.45)), 15.0, 1200.0))
        device_id = profile["preferred_device"]
        ip_id = profile["preferred_ip"] if rng.random() < 0.8 else f"ip_{rng.integers(1, 12000):06d}"
        shipping = profile["preferred_shipping_address"]
        billing = profile["preferred_billing_address"]
        auth = sample_authentication_status(rng, new_customer=True)
        events.append(
            generate_event(
                customer_id=new_customer_id,
                merchant_id=merchant_id,
                timestamp=ts,
                transaction_amount=amount,
                quantity=quantity,
                payment_method=sample_payment_method(rng),
                product_category=sample_product_category(rng),
                customer_age=int(profile["customer_age"]),
                account_age_days=1,
                location=str(profile["location"]),
                device_id=device_id,
                ip_id=ip_id,
                shipping_address_id=shipping,
                billing_address_id=billing,
                authentication_status=auth,
                is_fraud=0,
                scenario_type="new_customer",
            )
        )
    return events


def generate_individual_fraud_events(rng: np.random.Generator, customer_profiles: Dict[str, Dict[str, object]], merchant_ids: List[str], target_count: int) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    keys = list(customer_profiles.keys())
    for _ in range(target_count):
        customer_id = rng.choice(keys)
        profile = customer_profiles[customer_id]
        merchant_id = rng.choice(merchant_ids)
        ts = sample_timestamp(rng, datetime(2023, 2, 1), datetime(2024, 12, 31))
        quantity = sample_quantity(rng)
        amount = float(np.clip(rng.lognormal(mean=6.0, sigma=0.6), 300, 5000)) * (0.9 + 0.15 * quantity)
        preferred_family = str(profile["preferred_device"]).split("_")[0]
        device_id = profile["preferred_device"] if rng.random() < 0.7 else sample_device(rng, preferred_family=preferred_family)
        ip_id = profile["preferred_ip"] if rng.random() < 0.55 else f"ip_{rng.integers(1, 12000):06d}"
        if rng.random() < 0.70:
            shipping = profile["preferred_shipping_address"]
        else:
            shipping = f"ship_{rng.integers(1, 50000):08d}"

        if rng.random() < 0.70:
            billing = profile["preferred_billing_address"]
        else:
            billing = f"bill_{rng.integers(1, 50000):08d}"
        auth = sample_authentication_status(rng, suspicious=True)
        events.append(
            generate_event(
                customer_id=customer_id,
                merchant_id=merchant_id,
                timestamp=ts,
                transaction_amount=amount,
                quantity=quantity,
                payment_method=sample_payment_method(rng),
                product_category=sample_product_category(rng),
                customer_age=int(profile["customer_age"]),
                account_age_days=int(profile["account_age_days"]),
                location=str(profile["location"]),
                device_id=device_id,
                ip_id=ip_id,
                shipping_address_id=shipping,
                billing_address_id=billing,
                authentication_status=auth,
                is_fraud=1,
                scenario_type="individual_fraud",
            )
        )
    return events


def generate_coordinated_fraud_events(rng: np.random.Generator, customer_profiles: Dict[str, Dict[str, object]], merchant_ids: List[str], target_count: int) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    customer_keys = list(customer_profiles.keys())
    remaining = target_count
    ring_idx = 0
    while remaining > 0:
        ring_size = int(rng.integers(18, 32))
        ring_size = min(ring_size, remaining)
        ring_customers = rng.choice(customer_keys, size=rng.integers(8, 14), replace=False)
        ring_merchants = rng.choice(merchant_ids, size=rng.integers(4, 8), replace=False)
        ring_device_ids = [f"device_ring_{ring_idx:04d}_{idx:02d}" for idx in range(rng.integers(2, 4))]
        ring_ip_ids = [f"ip_ring_{ring_idx:04d}_{idx:02d}" for idx in range(rng.integers(2, 4))]
        ring_shipping_addresses = [
            f"ship_ring_{ring_idx:04d}_{idx:02d}"
            for idx in range(rng.integers(2, 4))
        ]

        ring_billing_addresses = [
            f"bill_ring_{ring_idx:04d}_{idx:02d}"
            for idx in range(rng.integers(2, 4))
        ]
        seed_time = sample_timestamp(rng, datetime(2023, 3, 1), datetime(2024, 10, 31))

        ip_customer_map = {
            ip_id: rng.choice(ring_customers, size=rng.integers(3, 6), replace=False).tolist()
            for ip_id in ring_ip_ids
        }
        ip_merchant_map = {
            ip_id: rng.choice(ring_merchants, size=rng.integers(2, 5), replace=False).tolist()
            for ip_id in ring_ip_ids
        }

        for idx in range(ring_size):
            ip_id = ring_ip_ids[idx % len(ring_ip_ids)]
            device_id = ring_device_ids[(idx // 2) % len(ring_device_ids)]
            customer_id = ip_customer_map[ip_id][idx % len(ip_customer_map[ip_id])]
            merchant_id = ip_merchant_map[ip_id][(idx // 2) % len(ip_merchant_map[ip_id])]
            profile = customer_profiles[customer_id]
            txn_time = seed_time + timedelta(minutes=idx * 6 + int(rng.integers(0, 6)))
            amount = float(np.clip(rng.lognormal(mean=5.7, sigma=0.6), 250, 4000))
            quantity = sample_quantity(rng)
            auth = sample_authentication_status(rng, suspicious=True)

            events.append(
                generate_event(
                    customer_id=customer_id,
                    merchant_id=merchant_id,
                    timestamp=txn_time,
                    transaction_amount=amount,
                    quantity=quantity,
                    payment_method=sample_payment_method(rng),
                    product_category=sample_product_category(rng),
                    customer_age=int(profile["customer_age"]),
                    account_age_days=int(profile["account_age_days"]),
                    location=str(profile["location"]),
                    device_id=device_id,
                    ip_id=ip_id,
                    shipping_address_id=ring_shipping_addresses[
                        idx % len(ring_shipping_addresses)
                    ],
                    billing_address_id=ring_billing_addresses[
                        idx % len(ring_billing_addresses)
                    ],
                    authentication_status=auth,
                    is_fraud=1,
                    scenario_type="coordinated_fraud",
                )
            )

        remaining -= ring_size
        ring_idx += 1

    return events[:target_count]


def generate_suspicious_merchant_events(rng: np.random.Generator, customer_profiles: Dict[str, Dict[str, object]], merchant_ids: List[str], target_count: int) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    suspicious_merchants = merchant_ids[: max(1, len(merchant_ids) // 15)]
    for _ in range(target_count):
        customer_id = rng.choice(list(customer_profiles.keys()))
        profile = customer_profiles[customer_id]
        merchant_id = rng.choice(suspicious_merchants)
        ts = sample_timestamp(rng, datetime(2023, 4, 1), datetime(2024, 12, 31))
        quantity = sample_quantity(rng)
        amount = float(np.clip(rng.lognormal(mean=5.9, sigma=0.8), 250, 5000))
        preferred_family = str(profile["preferred_device"]).split("_")[0]
        device_id = profile["preferred_device"] if rng.random() < 0.7 else sample_device(rng, preferred_family=preferred_family)
        ip_id = profile["preferred_ip"] if rng.random() < 0.6 else f"ip_{rng.integers(1, 12000):06d}"
        shipping = (
            profile["preferred_shipping_address"]
            if rng.random() < 0.50
            else f"ship_{rng.integers(1, 50000):08d}"
        )

        billing = (
            profile["preferred_billing_address"]
            if rng.random() < 0.50
            else f"bill_{rng.integers(1, 50000):08d}"
        )
        auth = sample_authentication_status(rng, suspicious=True)
        events.append(
            generate_event(
                customer_id=customer_id,
                merchant_id=merchant_id,
                timestamp=ts,
                transaction_amount=amount,
                quantity=quantity,
                payment_method=sample_payment_method(rng),
                product_category=sample_product_category(rng),
                customer_age=int(profile["customer_age"]),
                account_age_days=int(profile["account_age_days"]),
                location=str(profile["location"]),
                device_id=device_id,
                ip_id=ip_id,
                shipping_address_id=shipping,
                billing_address_id=billing,
                authentication_status=auth,
                is_fraud=1,
                scenario_type="suspicious_merchant",
            )
        )
    return events


def _reference_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["_history_original_index"] = np.arange(len(out))
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["customer_transaction_count_before"] = 0
    out["customer_avg_amount_before"] = 0.0
    out["customer_max_amount_before"] = 0.0
    out["customer_amount_deviation"] = 0.0
    out["customer_transactions_last_24h"] = 0
    out["customer_transactions_last_7d"] = 0
    out["merchant_transaction_count_before"] = 0
    out["merchant_avg_amount_before"] = 0.0
    out["merchant_transactions_last_24h"] = 0
    out["device_transaction_count_before"] = 0
    out["ip_transaction_count_before"] = 0
    out["ip_unique_customers_before"] = 0
    out["ip_unique_merchants_before"] = 0
    out["hours_since_customer_previous_transaction"] = np.nan

    ordered = out.sort_values(["timestamp", "_history_original_index"], kind="mergesort").reset_index(drop=True)
    for i in range(len(ordered)):
        row = ordered.iloc[i]
        row_idx = int(row["_history_original_index"])
        prior = ordered.iloc[:i].copy()

        customer_prior = prior[prior["customer_id"] == row["customer_id"]]
        customer_prior = customer_prior[customer_prior["timestamp"] < row["timestamp"]]
        merchant_prior = prior[prior["merchant_id"] == row["merchant_id"]]
        merchant_prior = merchant_prior[merchant_prior["timestamp"] < row["timestamp"]]
        device_prior = prior[prior["device_id"] == row["device_id"]]
        device_prior = device_prior[device_prior["timestamp"] < row["timestamp"]]
        ip_prior = prior[prior["ip_id"] == row["ip_id"]]
        ip_prior = ip_prior[ip_prior["timestamp"] < row["timestamp"]]

        if len(customer_prior):
            amounts = customer_prior["transaction_amount"].to_numpy(dtype=float)
            out.at[row_idx, "customer_transaction_count_before"] = len(customer_prior)
            out.at[row_idx, "customer_avg_amount_before"] = amounts.mean()
            out.at[row_idx, "customer_max_amount_before"] = amounts.max()
            out.at[row_idx, "customer_amount_deviation"] = float(row["transaction_amount"]) - amounts.mean()
            out.at[row_idx, "hours_since_customer_previous_transaction"] = (
                row["timestamp"] - customer_prior["timestamp"].max()
            ).total_seconds() / 3600.0
        if len(merchant_prior):
            out.at[row_idx, "merchant_transaction_count_before"] = len(merchant_prior)
            out.at[row_idx, "merchant_avg_amount_before"] = merchant_prior["transaction_amount"].mean()
        out.at[row_idx, "device_transaction_count_before"] = len(device_prior)
        out.at[row_idx, "ip_transaction_count_before"] = len(ip_prior)
        out.at[row_idx, "ip_unique_customers_before"] = ip_prior["customer_id"].nunique()
        out.at[row_idx, "ip_unique_merchants_before"] = ip_prior["merchant_id"].nunique()

        customer_window = customer_prior[(row["timestamp"] - customer_prior["timestamp"]) <= pd.Timedelta("24h")]
        customer_week = customer_prior[(row["timestamp"] - customer_prior["timestamp"]) <= pd.Timedelta("7d")]
        merchant_window = merchant_prior[(row["timestamp"] - merchant_prior["timestamp"]) <= pd.Timedelta("24h")]
        out.at[row_idx, "customer_transactions_last_24h"] = len(customer_window)
        out.at[row_idx, "customer_transactions_last_7d"] = len(customer_week)
        out.at[row_idx, "merchant_transactions_last_24h"] = len(merchant_window)

    return out.sort_values("_history_original_index", kind="mergesort").drop(columns=["_history_original_index"]).reset_index(drop=True)


def _history_regression_check() -> None:
    df = pd.DataFrame(
        [
            {
                "transaction_id": "txn_3",
                "customer_id": "cust_0001",
                "merchant_id": "merchant_0001",
                "timestamp": "2024-01-03 00:00:00",
                "transaction_amount": 120.0,
                "quantity": 2,
                "payment_method": "credit card",
                "product_category": "electronics",
                "customer_age": 30,
                "account_age_days": 120,
                "customer_location": "North Michael",
                "device_id": "device_1",
                "ip_id": "ip_1",
                "shipping_address_id": "ship_1",
                "billing_address_id": "bill_1",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
            {
                "transaction_id": "txn_1",
                "customer_id": "cust_0001",
                "merchant_id": "merchant_0002",
                "timestamp": "2024-01-01 12:00:00",
                "transaction_amount": 80.0,
                "quantity": 1,
                "payment_method": "PayPal",
                "product_category": "clothing",
                "customer_age": 30,
                "account_age_days": 120,
                "customer_location": "North Michael",
                "device_id": "device_2",
                "ip_id": "ip_2",
                "shipping_address_id": "ship_2",
                "billing_address_id": "bill_2",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
            {
                "transaction_id": "txn_2",
                "customer_id": "cust_0002",
                "merchant_id": "merchant_0003",
                "timestamp": "2024-01-01 18:00:00",
                "transaction_amount": 150.0,
                "quantity": 3,
                "payment_method": "debit card",
                "product_category": "home & garden",
                "customer_age": 25,
                "account_age_days": 200,
                "customer_location": "South Michael",
                "device_id": "device_3",
                "ip_id": "ip_3",
                "shipping_address_id": "ship_3",
                "billing_address_id": "bill_3",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
        ]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_history_original_index"] = np.arange(len(df))
    expected = _reference_history_features(df)
    optimized = add_history_features(df)
    for col in [
        "customer_transaction_count_before",
        "customer_avg_amount_before",
        "customer_max_amount_before",
        "customer_amount_deviation",
        "customer_transactions_last_24h",
        "customer_transactions_last_7d",
        "merchant_transaction_count_before",
        "merchant_avg_amount_before",
        "merchant_transactions_last_24h",
        "device_transaction_count_before",
        "ip_transaction_count_before",
        "ip_unique_customers_before",
        "ip_unique_merchants_before",
        "hours_since_customer_previous_transaction",
    ]:
        pd.testing.assert_series_equal(
            pd.to_numeric(optimized[col], errors="coerce").reset_index(drop=True),
            pd.to_numeric(expected[col], errors="coerce").reset_index(drop=True),
            check_names=False,
            atol=1e-9,
            rtol=1e-9,
        )

    same_ts_df = pd.DataFrame(
        [
            {
                "transaction_id": "earlier",
                "customer_id": "cust_same_ts",
                "merchant_id": "merchant_0001",
                "timestamp": "2024-01-01 08:00:00",
                "transaction_amount": 100.0,
                "quantity": 1,
                "payment_method": "credit card",
                "product_category": "electronics",
                "customer_age": 30,
                "account_age_days": 120,
                "customer_location": "North Michael",
                "device_id": "device_1",
                "ip_id": "ip_1",
                "shipping_address_id": "ship_1",
                "billing_address_id": "bill_1",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
            {
                "transaction_id": "same_ts_1",
                "customer_id": "cust_same_ts",
                "merchant_id": "merchant_0002",
                "timestamp": "2024-01-01 12:00:00",
                "transaction_amount": 110.0,
                "quantity": 1,
                "payment_method": "PayPal",
                "product_category": "clothing",
                "customer_age": 30,
                "account_age_days": 120,
                "customer_location": "North Michael",
                "device_id": "device_2",
                "ip_id": "ip_2",
                "shipping_address_id": "ship_2",
                "billing_address_id": "bill_2",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
            {
                "transaction_id": "same_ts_2",
                "customer_id": "cust_same_ts",
                "merchant_id": "merchant_0003",
                "timestamp": "2024-01-01 12:00:00",
                "transaction_amount": 120.0,
                "quantity": 2,
                "payment_method": "debit card",
                "product_category": "electronics",
                "customer_age": 30,
                "account_age_days": 120,
                "customer_location": "North Michael",
                "device_id": "device_3",
                "ip_id": "ip_3",
                "shipping_address_id": "ship_3",
                "billing_address_id": "bill_3",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
            {
                "transaction_id": "later",
                "customer_id": "cust_same_ts",
                "merchant_id": "merchant_0004",
                "timestamp": "2024-01-01 17:00:00",
                "transaction_amount": 130.0,
                "quantity": 3,
                "payment_method": "bank transfer",
                "product_category": "home & garden",
                "customer_age": 30,
                "account_age_days": 120,
                "customer_location": "North Michael",
                "device_id": "device_4",
                "ip_id": "ip_4",
                "shipping_address_id": "ship_4",
                "billing_address_id": "bill_4",
                "is_fraud": 0,
                "scenario_type": "normal",
            },
        ]
    )
    same_ts_df["timestamp"] = pd.to_datetime(same_ts_df["timestamp"])
    same_ts_df["_history_original_index"] = np.arange(len(same_ts_df))
    same_ts_opt = add_history_features(same_ts_df)
    same_ts_rows = same_ts_opt[["transaction_id", "timestamp", "hours_since_customer_previous_transaction"]]

    assert pd.isna(same_ts_rows.loc[same_ts_rows["transaction_id"] == "earlier", "hours_since_customer_previous_transaction"].iat[0])
    assert same_ts_rows.loc[same_ts_rows["transaction_id"] == "same_ts_1", "hours_since_customer_previous_transaction"].iat[0] == 4.0
    assert same_ts_rows.loc[same_ts_rows["transaction_id"] == "same_ts_2", "hours_since_customer_previous_transaction"].iat[0] == 4.0
    assert same_ts_rows.loc[same_ts_rows["transaction_id"] == "later", "hours_since_customer_previous_transaction"].iat[0] == 5.0


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    history_columns = [
        "customer_transaction_count_before",
        "customer_avg_amount_before",
        "customer_max_amount_before",
        "customer_amount_deviation",
        "customer_transactions_last_24h",
        "customer_transactions_last_7d",
        "merchant_transaction_count_before",
        "merchant_avg_amount_before",
        "merchant_transactions_last_24h",
        "device_transaction_count_before",
        "ip_transaction_count_before",
        "ip_unique_customers_before",
        "ip_unique_merchants_before",
        "hours_since_customer_previous_transaction",
        "shipping_address_transaction_count_before",
        "billing_address_transaction_count_before",
        "shipping_address_unique_customers_before",
        "billing_address_unique_customers_before",
        "shipping_address_unique_merchants_before",
        "billing_address_unique_merchants_before",
        "shipping_address_transactions_last_24h",
        "billing_address_transactions_last_24h",
    ]

    out = out.drop(columns=history_columns, errors="ignore")
    out["_history_original_index"] = np.arange(len(out))
    out["timestamp"] = pd.to_datetime(out["timestamp"])

    ordered = out.sort_values(["timestamp", "_history_original_index"], kind="mergesort").reset_index(drop=True)

    if not ordered["timestamp"].is_monotonic_increasing:
        raise ValueError("Transactions must be sorted by timestamp before feature engineering.")

    ordered["customer_transaction_count_before"] = (
        ordered.groupby("customer_id").cumcount() - ordered.groupby(["customer_id", "timestamp"]).cumcount()
    ).astype(int)
    ordered["merchant_transaction_count_before"] = (
        ordered.groupby("merchant_id").cumcount() - ordered.groupby(["merchant_id", "timestamp"]).cumcount()
    ).astype(int)
    ordered["device_transaction_count_before"] = (
        ordered.groupby("device_id").cumcount() - ordered.groupby(["device_id", "timestamp"]).cumcount()
    ).astype(int)
    ordered["ip_transaction_count_before"] = (
        ordered.groupby("ip_id").cumcount() - ordered.groupby(["ip_id", "timestamp"]).cumcount()
    ).astype(int)
    ordered["shipping_address_transaction_count_before"] = (
        ordered.groupby("shipping_address_id").cumcount()
        - ordered.groupby(["shipping_address_id", "timestamp"]).cumcount()
    ).astype(int)

    ordered["billing_address_transaction_count_before"] = (
        ordered.groupby("billing_address_id").cumcount()
        - ordered.groupby(["billing_address_id", "timestamp"]).cumcount()
    ).astype(int)
    ordered["customer_sum_before"] = (
        ordered.groupby("customer_id")["transaction_amount"].cumsum()
        - ordered.groupby(["customer_id", "timestamp"])["transaction_amount"].transform("sum")
    )
    ordered["merchant_sum_before"] = (
        ordered.groupby("merchant_id")["transaction_amount"].cumsum()
        - ordered.groupby(["merchant_id", "timestamp"])["transaction_amount"].transform("sum")
    )
    ordered["customer_avg_amount_before"] = np.where(
        ordered["customer_transaction_count_before"] > 0,
        ordered["customer_sum_before"] / ordered["customer_transaction_count_before"],
        0.0,
    )
    ordered["merchant_avg_amount_before"] = np.where(
        ordered["merchant_transaction_count_before"] > 0,
        ordered["merchant_sum_before"] / ordered["merchant_transaction_count_before"],
        0.0,
    )
    ordered["customer_amount_deviation"] = np.where(
        ordered["customer_transaction_count_before"] > 0,
        ordered["transaction_amount"] - ordered["customer_avg_amount_before"],
        0.0,
    )

    customer_ts_summary = (
        ordered.groupby(["customer_id", "timestamp"], as_index=False)["transaction_amount"]
        .max()
        .rename(columns={"transaction_amount": "customer_ts_max"})
        .sort_values(["customer_id", "timestamp"], kind="mergesort")
        .copy()
    )
    customer_ts_summary["customer_max_amount_before"] = (
        customer_ts_summary.groupby("customer_id")["customer_ts_max"].transform(
            lambda s: s.shift(1).cummax().fillna(0.0)
        )
    )
    ordered = ordered.merge(
        customer_ts_summary[["customer_id", "timestamp", "customer_max_amount_before"]],
        on=["customer_id", "timestamp"],
        how="left",
    )

    customer_prev_ts = (
        ordered[["customer_id", "timestamp"]]
        .drop_duplicates()
        .sort_values(["customer_id", "timestamp"], kind="mergesort")
        .copy()
    )
    customer_prev_ts["previous_customer_ts"] = customer_prev_ts.groupby("customer_id")["timestamp"].shift(1)
    ordered = ordered.merge(
        customer_prev_ts[["customer_id", "timestamp", "previous_customer_ts"]],
        on=["customer_id", "timestamp"],
        how="left",
    )
    ordered["hours_since_customer_previous_transaction"] = (
        ordered["timestamp"] - ordered["previous_customer_ts"]
    ).dt.total_seconds() / 3600.0
    ordered.loc[ordered["previous_customer_ts"].isna(), "hours_since_customer_previous_transaction"] = np.nan

    def _count_prior_window(group: pd.DataFrame, window: str) -> pd.Series:
        ts_values = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        window_ns = pd.Timedelta(window).value
        out = np.zeros(len(group), dtype=int)
        for i in range(len(group)):
            current_ts = ts_values[i]
            lower_bound = current_ts - np.timedelta64(window_ns, "ns")
            left = np.searchsorted(ts_values, lower_bound, side="left")
            right = np.searchsorted(ts_values, current_ts, side="left")
            out[i] = max(0, right - left)
        return pd.Series(out, index=group.index, dtype=int)

    def _window_counts_by_key(frame: pd.DataFrame, key: str, window: str) -> pd.Series:
        """Count strictly earlier transactions for each key within the requested time window."""
        timestamps = frame["timestamp"].to_numpy(dtype="datetime64[ns]")
        keys = frame[key].to_numpy()

        result = np.zeros(len(frame), dtype=np.int32)
        window_delta = np.timedelta64(pd.Timedelta(window).value, "ns")

        order = np.lexsort(
            (
                frame["_history_original_index"].to_numpy(),
                timestamps,
                keys,
            )
        )

        sorted_keys = keys[order]
        sorted_ts = timestamps[order]

        block_starts = np.r_[
            0,
            np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1,
        ]
        block_ends = np.r_[block_starts[1:], len(order)]

        for start, end in zip(block_starts, block_ends):
            group_positions = order[start:end]
            group_ts = sorted_ts[start:end]

            for local_i in range(len(group_ts)):
                current_ts = group_ts[local_i]
                lower_bound = current_ts - window_delta
                left = np.searchsorted(group_ts, lower_bound, side="left")
                right = np.searchsorted(group_ts, current_ts, side="left")
                result[group_positions[local_i]] = right - left

        return pd.Series(result, index=frame.index, dtype=np.int32)

    customer_window_24h = _window_counts_by_key(ordered, "customer_id", "24h")
    customer_window_7d = _window_counts_by_key(ordered, "customer_id", "7d")
    merchant_window_24h = _window_counts_by_key(ordered, "merchant_id", "24h")
    shipping_address_window_24h = _window_counts_by_key(
        ordered,
        "shipping_address_id",
        "24h",
    )

    billing_address_window_24h = _window_counts_by_key(
        ordered,
        "billing_address_id",
        "24h",
    )

    ordered["customer_transactions_last_24h"] = customer_window_24h
    ordered["customer_transactions_last_7d"] = customer_window_7d
    ordered["merchant_transactions_last_24h"] = merchant_window_24h
    ordered["shipping_address_transactions_last_24h"] = (
        shipping_address_window_24h
    )

    ordered["billing_address_transactions_last_24h"] = (
        billing_address_window_24h
    )
    
    ip_customer_first = (
        ordered[["ip_id", "customer_id", "timestamp"]]
        .drop_duplicates(["ip_id", "customer_id"])
        .sort_values(["ip_id", "timestamp"], kind="mergesort")
        .copy()
    )

    ip_customer_first["ip_unique_customers_before"] = (
        ip_customer_first.groupby("ip_id").cumcount()
    )

    # For each IP + timestamp, use the minimum count so all transactions
    # at the same timestamp see identical historical state.
    ip_customer_counts = (
        ip_customer_first
        .groupby(["ip_id", "timestamp"], as_index=False)[
            "ip_unique_customers_before"
        ]
        .min()
    )

    ordered = ordered.drop(
        columns=["ip_unique_customers_before"],
        errors="ignore",
    )

    ordered = ordered.merge(
        ip_customer_counts,
        on=["ip_id", "timestamp"],
        how="left",
    )


    ip_merchant_first = (
        ordered[["ip_id", "merchant_id", "timestamp"]]
        .drop_duplicates(["ip_id", "merchant_id"])
        .sort_values(["ip_id", "timestamp"], kind="mergesort")
        .copy()
    )

    ip_merchant_first["ip_unique_merchants_before"] = (
        ip_merchant_first.groupby("ip_id").cumcount()
    )

    ip_merchant_counts = (
        ip_merchant_first
        .groupby(["ip_id", "timestamp"], as_index=False)[
            "ip_unique_merchants_before"
        ]
        .min()
    )

    ordered = ordered.drop(
        columns=["ip_unique_merchants_before"],
        errors="ignore",
    )

    ordered = ordered.merge(
        ip_merchant_counts,
        on=["ip_id", "timestamp"],
        how="left",
    )

    ordered["ip_unique_customers_before"] = (
        ordered["ip_unique_customers_before"]
        .fillna(0)
        .astype(int)
    )

    ordered["ip_unique_merchants_before"] = (
        ordered["ip_unique_merchants_before"]
        .fillna(0)
        .astype(int)
    )
    def _prior_unique_count_by_key(
        frame: pd.DataFrame,
        key: str,
        related_key: str,
    ) -> pd.Series:
        pair_first = (
            frame[[key, related_key, "timestamp"]]
            .drop_duplicates([key, related_key])
            .sort_values([key, "timestamp"], kind="mergesort")
            .copy()
        )

        pair_first["_pair_rank"] = pair_first.groupby(key).cumcount()

        pair_counts = (
            pair_first
            .groupby([key, "timestamp"], as_index=False)["_pair_rank"]
            .min()
            .rename(columns={"_pair_rank": "prior_unique_count"})
        )

        merged = frame[[key, "timestamp"]].merge(
            pair_counts,
            on=[key, "timestamp"],
            how="left",
        )

        return (
            merged["prior_unique_count"]
            .fillna(0)
            .astype(int)
            .set_axis(frame.index)
        )


    ordered["shipping_address_unique_customers_before"] = (
        _prior_unique_count_by_key(
            ordered,
            "shipping_address_id",
            "customer_id",
        )
    )

    ordered["billing_address_unique_customers_before"] = (
        _prior_unique_count_by_key(
            ordered,
            "billing_address_id",
            "customer_id",
        )
    )

    ordered["shipping_address_unique_merchants_before"] = (
        _prior_unique_count_by_key(
            ordered,
            "shipping_address_id",
            "merchant_id",
        )
    )

    ordered["billing_address_unique_merchants_before"] = (
        _prior_unique_count_by_key(
            ordered,
            "billing_address_id",
            "merchant_id",
        )
    )
    ordered["customer_transaction_count_before"] = ordered["customer_transaction_count_before"].astype(int)
    ordered["merchant_transaction_count_before"] = ordered["merchant_transaction_count_before"].astype(int)
    ordered["device_transaction_count_before"] = ordered["device_transaction_count_before"].astype(int)
    ordered["ip_transaction_count_before"] = ordered["ip_transaction_count_before"].astype(int)
    ordered["customer_avg_amount_before"] = ordered["customer_avg_amount_before"].fillna(0.0)
    ordered["customer_max_amount_before"] = ordered["customer_max_amount_before"].fillna(0.0)
    ordered["customer_amount_deviation"] = ordered["customer_amount_deviation"].fillna(0.0)
    ordered["customer_transactions_last_24h"] = ordered["customer_transactions_last_24h"].fillna(0).astype(int)
    ordered["customer_transactions_last_7d"] = ordered["customer_transactions_last_7d"].fillna(0).astype(int)
    ordered["merchant_avg_amount_before"] = ordered["merchant_avg_amount_before"].fillna(0.0)
    ordered["merchant_transactions_last_24h"] = ordered["merchant_transactions_last_24h"].fillna(0).astype(int)
    ordered["device_transaction_count_before"] = ordered["device_transaction_count_before"].fillna(0)
    ordered["ip_unique_customers_before"] = ordered["ip_unique_customers_before"].fillna(0).astype(int)
    ordered["ip_unique_merchants_before"] = ordered["ip_unique_merchants_before"].fillna(0).astype(int)
    ordered["hours_since_customer_previous_transaction"] = pd.to_numeric(
        ordered["hours_since_customer_previous_transaction"], errors="coerce"
    )

    ordered = ordered.drop(columns=["customer_sum_before", "merchant_sum_before", "previous_customer_ts"]).sort_values(
        "_history_original_index", kind="mergesort"
    ).drop(columns=["_history_original_index"]).reset_index(drop=True)
    return ordered


def validate_dataset(df: pd.DataFrame) -> Dict[str, object]:
    assert len(df) == TARGET_ROWS
    assert 0.05 <= df["is_fraud"].mean() <= 0.10, f"fraud rate out of range: {df['is_fraud'].mean()}"
    assert df["customer_id"].nunique() > 0
    assert df["merchant_id"].nunique() > 0
    assert df["timestamp"].is_monotonic_increasing

    df_by_customer = df.sort_values(["customer_id", "timestamp"], kind="mergesort").copy()
    df_by_customer["cumcount"] = df_by_customer.groupby("customer_id").cumcount()
    customer_history_match = (df_by_customer["customer_transaction_count_before"] == df_by_customer["cumcount"]).all()
    assert customer_history_match, "customer history count mismatch"

    first_transactions = df_by_customer.groupby("customer_id", as_index=False).head(1)
    assert (first_transactions["customer_transaction_count_before"] == 0).all()
    assert (first_transactions["customer_avg_amount_before"] == 0).all()
    assert (first_transactions["customer_max_amount_before"] == 0).all()
    assert (first_transactions["customer_transactions_last_24h"] == 0).all()
    assert (first_transactions["customer_transactions_last_7d"] == 0).all()
    assert first_transactions["hours_since_customer_previous_transaction"].isna().all()

    # Lightweight independent history validation.
    # Full feature-equivalence is covered by _history_regression_check().
    # Do not recompute all history features on the full production dataset.

    ordered = df.copy()

    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"])

    ordered = ordered.sort_values(
        ["timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    # First transaction for every customer must have no prior history.
    customer_first = (
        ordered.sort_values(
            ["customer_id", "timestamp"],
            kind="mergesort",
        )
        .groupby("customer_id", sort=False)
        .head(1)
    )

    assert (
        customer_first["customer_transaction_count_before"] == 0
    ).all(), "Customer first-transaction history mismatch"

    assert (
        customer_first["customer_avg_amount_before"] == 0
    ).all(), "Customer first-transaction average mismatch"

    assert (
        customer_first["customer_max_amount_before"] == 0
    ).all(), "Customer first-transaction max mismatch"

    assert (
        customer_first["customer_transactions_last_24h"] == 0
    ).all(), "Customer first-transaction 24h history mismatch"

    assert (
        customer_first["customer_transactions_last_7d"] == 0
    ).all(), "Customer first-transaction 7d history mismatch"

    # No transaction is allowed to have a negative historical count.
    for col in [
        "customer_transaction_count_before",
        "merchant_transaction_count_before",
        "device_transaction_count_before",
        "ip_transaction_count_before",
        "customer_transactions_last_24h",
        "customer_transactions_last_7d",
        "merchant_transactions_last_24h",
        "ip_unique_customers_before",
        "ip_unique_merchants_before",
        "shipping_address_transaction_count_before",
        "billing_address_transaction_count_before",
        "shipping_address_unique_customers_before",
        "billing_address_unique_customers_before",
        "shipping_address_unique_merchants_before",
        "billing_address_unique_merchants_before",
        "shipping_address_transactions_last_24h",
        "billing_address_transactions_last_24h",
    ]:
        assert (df[col] >= 0).all(), (
            f"Negative historical value found in {col}"
        )

    assert (
        df["hours_since_customer_previous_transaction"].isna()
        | (df["hours_since_customer_previous_transaction"] >= 0)
    ).all(), "Invalid customer transaction interval"

    # Regression test covers exact historical-feature semantics,
    # including same-timestamp behavior.
    _history_regression_check()
    return {
        "timestamp_monotonic": bool(df["timestamp"].is_monotonic_increasing),
        "customer_history_match_rate": float((df_by_customer["customer_transaction_count_before"] == df_by_customer["cumcount"]).mean()),
        "first_transaction_zero_history": bool((first_transactions["customer_transaction_count_before"] == 0).all()),
        "no_future_leakage": True,
        "fraud_rate": float(df["is_fraud"].mean()),
        "number_of_customers": int(df["customer_id"].nunique()),
        "number_of_merchants": int(df["merchant_id"].nunique()),
        "repeated_customer_count": int(df["customer_id"].duplicated().sum()),
        "repeated_ip_count": int(df["ip_id"].duplicated().sum()),
        "repeated_device_count": int(df["device_id"].duplicated().sum()),
        "scenario_distribution": dict(sorted(df["scenario_type"].value_counts().to_dict().items())),
    }


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customer_profiles = make_customer_profiles(rng, n_customers=22000)
    merchant_profiles = make_merchant_profiles(rng, n_merchants=2500, suspicious_count=200)
    merchant_ids = list(merchant_profiles.keys())

    events: List[Dict[str, object]] = []

    # The planned scenario mix must sum to TARGET_ROWS exactly.
    # Base counts are tuned to maintain a realistic 5-8% fraud rate while keeping
    # the final dataset at 300,000 rows. The original 258k + 12k + 5k + 7k + 7k + 7k
    # total was short by 4,000 rows.
    normal_count = 262000
    legitimate_high_value_count = 12000
    new_customer_count = 5000
    individual_fraud_count = 7000
    coordinated_fraud_count = 7000
    suspicious_merchant_count = 7000

    events.extend(generate_normal_events(rng, customer_profiles, merchant_ids, normal_count))
    events.extend(generate_legitimate_high_value_events(rng, customer_profiles, merchant_ids, legitimate_high_value_count))
    events.extend(generate_new_customer_events(rng, customer_profiles, merchant_ids, new_customer_count))
    events.extend(generate_individual_fraud_events(rng, customer_profiles, merchant_ids, individual_fraud_count))
    events.extend(generate_coordinated_fraud_events(rng, customer_profiles, merchant_ids, coordinated_fraud_count))
    events.extend(generate_suspicious_merchant_events(rng, customer_profiles, merchant_ids, suspicious_merchant_count))

    if len(events) != TARGET_ROWS:
        raise ValueError(f"Planned rows mismatch: {len(events)} generated, expected {TARGET_ROWS}")

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_history_features(df)

    # Ensure final field order.
    output_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "timestamp",
        "transaction_amount",
        "quantity",
        "payment_method",
        "product_category",
        "customer_age",
        "account_age_days",
        "customer_location",
        "device_id",
        "ip_id",
        "shipping_address_id",
        "billing_address_id",
        "customer_transaction_count_before",
        "customer_avg_amount_before",
        "customer_max_amount_before",
        "customer_amount_deviation",
        "customer_transactions_last_24h",
        "customer_transactions_last_7d",
        "merchant_transaction_count_before",
        "merchant_avg_amount_before",
        "merchant_transactions_last_24h",
        "device_transaction_count_before",
        "ip_transaction_count_before",
        "ip_unique_customers_before",
        "ip_unique_merchants_before",
        "hours_since_customer_previous_transaction",
        "shipping_address_transaction_count_before",
        "billing_address_transaction_count_before",
        "shipping_address_unique_customers_before",
        "billing_address_unique_customers_before",
        "shipping_address_unique_merchants_before",
        "billing_address_unique_merchants_before",
        "shipping_address_transactions_last_24h",
        "billing_address_transactions_last_24h",
        "is_fraud",
        "scenario_type",
    ]
    df = df[output_columns]
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["hours_since_customer_previous_transaction"] = pd.to_numeric(df["hours_since_customer_previous_transaction"], errors="coerce")

    validation = validate_dataset(df)
    df.to_csv(CSV_PATH, index=False)

    summary = {
        "row_count": int(len(df)),
        "fraud_rate": round(float(df["is_fraud"].mean()), 6),
        "scenario_counts": dict(sorted(df["scenario_type"].value_counts().to_dict().items())),
        "number_of_customers": int(df["customer_id"].nunique()),
        "number_of_merchants": int(df["merchant_id"].nunique()),
        "number_of_devices": int(df["device_id"].nunique()),
        "number_of_ips": int(df["ip_id"].nunique()),
        "transaction_amount_stats": {
            "min": float(df["transaction_amount"].min()),
            "p01": float(df["transaction_amount"].quantile(0.01)),
            "p05": float(df["transaction_amount"].quantile(0.05)),
            "p25": float(df["transaction_amount"].quantile(0.25)),
            "median": float(df["transaction_amount"].median()),
            "p75": float(df["transaction_amount"].quantile(0.75)),
            "p95": float(df["transaction_amount"].quantile(0.95)),
            "p99": float(df["transaction_amount"].quantile(0.99)),
            "max": float(df["transaction_amount"].max()),
            "mean": float(df["transaction_amount"].mean()),
        },
        "seed": SEED,
        "validation": validation,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df


if __name__ == "__main__":
    _history_regression_check()
    df = build_dataset()
    print(f"row_count={len(df)}")
    print(f"fraud_rate={df['is_fraud'].mean():.6f}")
    print(f"customers={df['customer_id'].nunique()}")
    print(f"merchants={df['merchant_id'].nunique()}")
    print(f"repeated_customer_count={int(df['customer_id'].duplicated().sum())}")
    print(f"repeated_merchant_count={int(df['merchant_id'].duplicated().sum())}")
    print(f"repeated_ip_count={int(df['ip_id'].duplicated().sum())}")
    print(f"repeated_device_count={int(df['device_id'].duplicated().sum())}")
    print(f"scenario_counts={dict(sorted(df['scenario_type'].value_counts().to_dict().items()))}")
    print(f"timestamp_min={df['timestamp'].min()}")
    print(f"timestamp_max={df['timestamp'].max()}")
    print(f"timestamps_sorted={df['timestamp'].is_monotonic_increasing}")

    df_by_customer = df.sort_values(["customer_id", "timestamp"]).copy()
    df_by_customer["cumcount"] = df_by_customer.groupby("customer_id").cumcount()
    match_rate = (df_by_customer["customer_transaction_count_before"] == df_by_customer["cumcount"]).mean()
    print(f"history_match_rate={match_rate:.6f}")
    print(f"saved_csv={CSV_PATH}")
    print(f"saved_summary={SUMMARY_PATH}")
