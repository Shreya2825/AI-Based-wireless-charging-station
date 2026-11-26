"""
=============================================================================
AI-Based Smart EV Charging Station — Dataset Generator
=============================================================================
Generates a realistic 50,000+ row dataset with correlated features,
noise, and outliers to simulate real-world EV charging scenarios.

Columns:
    battery_level, charging_cycles, temperature, voltage,
    current_load, grid_demand, energy_consumed, charging_time,
    time_of_day, day_of_week, station_id,
    user_lat, user_lon, station_lat, station_lon,
    battery_health, priority, price_per_kwh,
    is_overheating (safety label), thermal_risk_score
=============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TOTAL_ROWS = 52_000          # slightly above 50k for safety
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Station geographic coordinates (10 stations in a metro area)
STATIONS = [
    {"id": f"STN_{i:02d}", "lat": 28.60 + np.random.uniform(-0.15, 0.15),
     "lon": 77.20 + np.random.uniform(-0.15, 0.15),
     "capacity_kw": np.random.choice([50, 100, 150, 250])}
    for i in range(1, 11)
]

OUTPUT_FILE = "ev_dataset.csv"


# ─────────────────────────────────────────────
# HELPER: Time-of-day demand multiplier
# ─────────────────────────────────────────────
def demand_multiplier(hour: int) -> float:
    """Simulate demand peaks at morning and evening rush hours."""
    if 7 <= hour <= 9:      # Morning rush
        return np.random.uniform(1.4, 1.8)
    elif 17 <= hour <= 20:  # Evening rush
        return np.random.uniform(1.5, 2.0)
    elif 23 <= hour or hour <= 5:  # Night (cheap)
        return np.random.uniform(0.4, 0.7)
    else:
        return np.random.uniform(0.8, 1.2)


# ─────────────────────────────────────────────
# MAIN DATA GENERATION
# ─────────────────────────────────────────────
def generate_dataset(n: int = TOTAL_ROWS) -> pd.DataFrame:
    print(f"[INFO] Generating {n:,} rows of EV charging data …")

    # ── Time features ──────────────────────────────────────────────────────
    base_time = datetime(2023, 1, 1)
    timestamps = [base_time + timedelta(minutes=random.randint(0, 525_600))
                  for _ in range(n)]
    time_of_day  = np.array([t.hour for t in timestamps])
    day_of_week  = np.array([t.weekday() for t in timestamps])   # 0=Mon
    month        = np.array([t.month for t in timestamps])

    # ── Battery features (correlated) ──────────────────────────────────────
    # battery_level: 5–100%
    battery_level = np.clip(
        np.random.beta(2, 2, n) * 100 + np.random.normal(0, 5, n), 5, 100
    )

    # charging_cycles: high cycles → worse health
    charging_cycles = np.random.gamma(shape=3, scale=120, size=n).astype(int)
    charging_cycles = np.clip(charging_cycles, 0, 1500)

    # battery_health: correlated with cycles & temperature
    battery_health_base = 100 - (charging_cycles / 1500) * 40
    battery_health = np.clip(
        battery_health_base + np.random.normal(0, 3, n), 40, 100
    )

    # ── Environmental / electrical features ───────────────────────────────
    temperature = np.clip(
        np.random.normal(30, 10, n) + (battery_level < 20) * 5, 10, 75
    )

    voltage = np.clip(
        np.random.normal(400, 20, n) - (battery_health < 60) * 15
        + np.random.normal(0, 8, n), 300, 480
    )

    # ── Load & Grid features ───────────────────────────────────────────────
    dm = np.array([demand_multiplier(h) for h in time_of_day])

    grid_demand = np.clip(
        np.random.normal(5000, 800, n) * dm + np.random.normal(0, 200, n),
        1000, 12000
    )

    current_load = np.clip(
        grid_demand * np.random.uniform(0.3, 0.8, n)
        + np.random.normal(0, 100, n), 200, 10000
    )

    energy_consumed = np.clip(
        (100 - battery_level) / 100 * np.random.uniform(20, 80, n)
        + np.random.normal(0, 2, n), 1, 100
    )

    charging_time = np.clip(
        energy_consumed / (current_load / 1000 + 0.01)
        + np.random.normal(0, 5, n), 5, 300
    )

    # ── Station assignment ─────────────────────────────────────────────────
    station_idx  = np.random.randint(0, len(STATIONS), n)
    station_ids  = [STATIONS[i]["id"] for i in station_idx]
    station_lats = np.array([STATIONS[i]["lat"] for i in station_idx])
    station_lons = np.array([STATIONS[i]["lon"] for i in station_idx])

    # User locations: within ~20 km of station
    user_lat = station_lats + np.random.uniform(-0.15, 0.15, n)
    user_lon = station_lons + np.random.uniform(-0.15, 0.15, n)

    # ── Dynamic Pricing ────────────────────────────────────────────────────
    price_base = 6.0   # ₹ per kWh base
    price_per_kwh = np.clip(
        price_base * dm
        + (grid_demand / 12000) * 4
        + (battery_health < 60) * 1.5
        + np.random.normal(0, 0.5, n), 3.0, 20.0
    )

    # ── Priority Score (AI target) ─────────────────────────────────────────
    # Higher score = higher priority for charging slot
    priority_score = (
        (100 - battery_level) * 0.35
        + (100 - battery_health) * 0.25
        + charging_cycles / 1500 * 20
        + (temperature > 45) * 10
        + dm * 5
        + np.random.normal(0, 3, n)
    )
    priority_score = np.clip(priority_score, 0, 100)
    priority_label = pd.cut(
        priority_score,
        bins=[0, 33, 66, 100],
        labels=["Low", "Medium", "High"]
    )

    # ── Safety / Thermal Module ────────────────────────────────────────────
    thermal_risk_score = np.clip(
        (temperature - 25) / 50 * 50
        + (voltage > 440) * 10
        + (battery_health < 60) * 15
        + (charging_cycles > 1000) * 10
        + np.random.normal(0, 2, n), 0, 100
    )
    is_overheating = (thermal_risk_score > 70).astype(int)

    # ── Carbon Emission (kg CO2 per session) ──────────────────────────────
    # Indian grid emission factor ≈ 0.82 kg CO2 / kWh
    carbon_emission_kg = np.round(energy_consumed * 0.82 * np.random.uniform(0.9, 1.1, n), 3)

    # ── Wireless Charging Efficiency ──────────────────────────────────────
    # Decreases with distance and poor alignment (simulated 0–10 cm gap)
    air_gap_cm = np.random.uniform(0, 10, n)
    wireless_efficiency_pct = np.clip(
        95 - air_gap_cm * 2.5 - (temperature > 40) * 5
        + np.random.normal(0, 2, n), 60, 98
    )

    # ── V2G Energy (kWh fed back to grid) ─────────────────────────────────
    v2g_active = ((battery_level > 80) & (grid_demand > 8000)).astype(int)
    v2g_energy_kwh = np.where(
        v2g_active,
        np.clip(np.random.uniform(2, 15, n), 0, 30),
        0.0
    )

    # ── Inject outliers (~2%) ─────────────────────────────────────────────
    outlier_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    temperature[outlier_idx] = np.random.uniform(65, 80, len(outlier_idx))
    voltage[outlier_idx]     = np.random.choice([250, 490], len(outlier_idx))
    is_overheating[outlier_idx] = 1

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":              [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "time_of_day":            time_of_day,
        "day_of_week":            day_of_week,
        "month":                  month,
        "station_id":             station_ids,
        "user_lat":               np.round(user_lat, 6),
        "user_lon":               np.round(user_lon, 6),
        "station_lat":            np.round(station_lats, 6),
        "station_lon":            np.round(station_lons, 6),
        "battery_level":          np.round(battery_level, 2),
        "charging_cycles":        charging_cycles,
        "temperature":            np.round(temperature, 2),
        "voltage":                np.round(voltage, 2),
        "current_load":           np.round(current_load, 2),
        "grid_demand":            np.round(grid_demand, 2),
        "energy_consumed":        np.round(energy_consumed, 3),
        "charging_time":          np.round(charging_time, 2),
        "air_gap_cm":             np.round(air_gap_cm, 2),
        "wireless_efficiency_pct": np.round(wireless_efficiency_pct, 2),
        "v2g_active":             v2g_active,
        "v2g_energy_kwh":         np.round(v2g_energy_kwh, 3),
        "battery_health":         np.round(battery_health, 2),
        "priority":               priority_label,
        "price_per_kwh":          np.round(price_per_kwh, 3),
        "thermal_risk_score":     np.round(thermal_risk_score, 2),
        "is_overheating":         is_overheating,
        "carbon_emission_kg":     carbon_emission_kg,
    })

    # Drop rare NaN in priority label (edge bins)
    df.dropna(subset=["priority"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Overheating events: {df['is_overheating'].sum():,} ({df['is_overheating'].mean()*100:.1f}%)")
    print(f"[INFO] Priority distribution:\n{df['priority'].value_counts()}")
    print(f"[INFO] Saving to '{OUTPUT_FILE}' …")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Dataset saved: {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("\n[SAMPLE]\n", df.head(3).to_string())
