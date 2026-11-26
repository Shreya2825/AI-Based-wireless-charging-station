"""
=============================================================================
AI-Based Smart EV Charging Station — Flask Backend (app.py)
=============================================================================
REST API endpoints:
  POST /predict_health      — Battery health classification
  POST /predict_load        — Current load prediction
  POST /predict_price       — Dynamic price prediction
  POST /predict_priority    — Charging priority classification
  POST /predict_thermal     — Thermal risk / overheating detection
  POST /nearest_station     — Nearest station (Haversine)
  POST /wireless_efficiency — Wireless charging efficiency
  GET  /live_data           — Simulated real-time dashboard data
  GET  /stations            — All station info
  POST /chatbot             — AI chatbot for EV users
  GET  /carbon_stats        — Carbon emission analytics
=============================================================================
Run:  python app.py
=============================================================================
"""

import os, math, random, json, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

MODELS_FILE   = "saved_models.pkl"
STATIONS_FILE = "stations_info.csv"

# ── Load models at startup ───────────────────────────────────────────────────
models = {}
stations_df = pd.DataFrame()

def load_artifacts():
    global models, stations_df
    if os.path.exists(MODELS_FILE):
        models = joblib.load(MODELS_FILE)
        print(f"[OK] Loaded models: {list(models.keys())}")
    else:
        print(f"[WARN] {MODELS_FILE} not found — run model_training.py first")

    if os.path.exists(STATIONS_FILE):
        stations_df = pd.read_csv(STATIONS_FILE)
        print(f"[OK] Loaded {len(stations_df)} stations")
    else:
        # Fallback: generate synthetic stations
        stations_df = pd.DataFrame([{
            "station_id": f"STN_{i:02d}",
            "lat": 28.60 + random.uniform(-0.15, 0.15),
            "lon": 77.20 + random.uniform(-0.15, 0.15),
            "avg_load": random.uniform(500, 4000),
            "avg_price": random.uniform(5, 15),
            "sessions": random.randint(100, 5000)
        } for i in range(1, 11)])
        print("[WARN] stations_info.csv not found — using synthetic stations")

load_artifacts()

# ── Haversine Distance ───────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Feature helpers ──────────────────────────────────────────────────────────
def cyclic(val, period):
    return math.sin(2*math.pi*val/period), math.cos(2*math.pi*val/period)

def safe_predict(model_key, feature_names, data_dict):
    """Build feature array and run model prediction safely."""
    if model_key not in models:
        return None, "Model not loaded"
    m = models[model_key]
    try:
        row = [[data_dict.get(f, 0) for f in feature_names]]
        X = pd.DataFrame(row, columns=feature_names)
        return m["model"].predict(X), None
    except Exception as e:
        return None, str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── 1. Battery Health Prediction ─────────────────────────────────────────────
@app.route("/predict_health", methods=["POST"])
def predict_health():
    d = request.get_json(force=True)
    required = ["battery_level","charging_cycles","temperature","voltage",
                "energy_consumed","thermal_risk_score"]
    for r in required:
        if r not in d:
            return jsonify({"error": f"Missing field: {r}"}), 400

    hour = d.get("time_of_day", datetime.now().hour)
    hs, hc = cyclic(hour, 24)
    d["hour_sin"] = hs
    d["hour_cos"] = hc
    d["health_cycle_ratio"] = d.get("battery_health", 80) / (d["charging_cycles"] + 1)
    d["temp_voltage_interaction"] = d["temperature"] * d["voltage"]

    feat = models.get("battery_health_classifier", {}).get("features", [])
    pred, err = safe_predict("battery_health_classifier", feat, d)
    if err:
        return jsonify({"error": err}), 500

    label = "Good" if int(pred[0]) == 1 else "Poor"
    proba = None
    try:
        proba_arr = models["battery_health_classifier"]["model"].predict_proba(
            pd.DataFrame([[d.get(f, 0) for f in feat]], columns=feat)
        )
        proba = float(proba_arr[0][int(pred[0])])
    except Exception:
        pass

    return jsonify({
        "battery_health_status": label,
        "prediction_code": int(pred[0]),
        "confidence": round(proba * 100, 2) if proba else None,
        "recommendation": (
            "Battery is healthy. Continue normal charging." if label == "Good"
            else "Battery health is poor. Consider battery service or replacement."
        )
    })


# ── 2. Load Prediction ───────────────────────────────────────────────────────
@app.route("/predict_load", methods=["POST"])
def predict_load():
    d = request.get_json(force=True)
    hour = d.get("time_of_day", datetime.now().hour)
    dow  = d.get("day_of_week", datetime.now().weekday())
    hs, hc = cyclic(hour, 24)
    ds, dc = cyclic(dow, 7)
    d.update({"hour_sin": hs, "hour_cos": hc, "dow_sin": ds, "dow_cos": dc})

    feat = models.get("load_predictor", {}).get("features", [])
    pred, err = safe_predict("load_predictor", feat, d)
    if err:
        return jsonify({"error": err}), 500

    load_kw = float(pred[0])
    status = "Critical" if load_kw > 8000 else ("High" if load_kw > 5000 else "Normal")
    return jsonify({
        "predicted_load_kw": round(load_kw, 2),
        "load_status": status,
        "grid_stress_pct": round(min(load_kw / 12000 * 100, 100), 1)
    })


# ── 3. Price Prediction ──────────────────────────────────────────────────────
@app.route("/predict_price", methods=["POST"])
def predict_price():
    d = request.get_json(force=True)
    hour = d.get("time_of_day", datetime.now().hour)
    dow  = d.get("day_of_week", datetime.now().weekday())
    hs, hc = cyclic(hour, 24)
    ds, dc = cyclic(dow, 7)
    d.update({"hour_sin": hs, "hour_cos": hc, "dow_sin": ds, "dow_cos": dc})
    d["load_ratio"] = d.get("current_load", 2000) / (d.get("grid_demand", 5000) + 1)

    feat = models.get("price_predictor", {}).get("features", [])
    pred, err = safe_predict("price_predictor", feat, d)
    if err:
        return jsonify({"error": err}), 500

    price = float(pred[0])
    tier = "Peak" if price > 12 else ("Standard" if price > 7 else "Off-Peak")
    return jsonify({
        "price_per_kwh": round(max(price, 2.0), 3),
        "currency": "INR",
        "pricing_tier": tier,
        "best_time_to_charge": "11 PM – 5 AM (Off-Peak rates)"
    })


# ── 4. Priority Classification ───────────────────────────────────────────────
@app.route("/predict_priority", methods=["POST"])
def predict_priority():
    d = request.get_json(force=True)
    hour = d.get("time_of_day", datetime.now().hour)
    hs, hc = cyclic(hour, 24)
    d["hour_sin"] = hs
    d["hour_cos"] = hc
    d["state_of_charge_delta"] = 100 - d.get("battery_level", 50)

    feat = models.get("priority_classifier", {}).get("features", [])
    pred, err = safe_predict("priority_classifier", feat, d)
    if err:
        return jsonify({"error": err}), 500

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    label = label_map.get(int(pred[0]), "Medium")
    return jsonify({
        "charging_priority": label,
        "priority_code": int(pred[0]),
        "estimated_wait_minutes": {
            "High": random.randint(0, 5),
            "Medium": random.randint(10, 25),
            "Low": random.randint(30, 60)
        }[label]
    })


# ── 5. Thermal Risk / Overheating Detection ──────────────────────────────────
@app.route("/predict_thermal", methods=["POST"])
def predict_thermal():
    d = request.get_json(force=True)
    d["temp_voltage_interaction"] = d.get("temperature", 30) * d.get("voltage", 400)

    # Thermal risk score
    feat_t = models.get("thermal_risk_regressor", {}).get("features", [])
    pred_t, err_t = safe_predict("thermal_risk_regressor", feat_t, d)
    risk_score = float(pred_t[0]) if pred_t is not None else 0.0

    # Overheating binary
    d["thermal_risk_score"] = risk_score
    feat_o = models.get("overheating_detector", {}).get("features", [])
    pred_o, err_o = safe_predict("overheating_detector", feat_o, d)
    is_overheat = int(pred_o[0]) if pred_o is not None else 0

    temp = d.get("temperature", 30)
    # Hard-coded safety cutoff
    auto_cutoff = temp > 65 or risk_score > 75

    alerts = []
    if auto_cutoff:
        alerts.append("🚨 EMERGENCY: Auto cut-off triggered! Battery temperature critical.")
    if temp > 50:
        alerts.append("⚠️ High temperature detected. Reduce charging rate.")
    if risk_score > 60:
        alerts.append("⚠️ Elevated thermal risk. Monitor closely.")

    return jsonify({
        "thermal_risk_score": round(risk_score, 2),
        "is_overheating": bool(is_overheat),
        "auto_cutoff_activated": auto_cutoff,
        "temperature_c": temp,
        "safety_status": "DANGER" if auto_cutoff else ("WARNING" if is_overheat else "SAFE"),
        "alerts": alerts,
        "thermal_runaway_info": (
            "Thermal runaway occurs when internal cell temperature rises uncontrollably, "
            "causing chain reactions leading to fire/explosion. "
            "AI monitors temperature, voltage, and charge cycles to prevent this."
        )
    })


# ── 6. Nearest Station ───────────────────────────────────────────────────────
@app.route("/nearest_station", methods=["POST"])
def nearest_station():
    d = request.get_json(force=True)
    user_lat = float(d.get("user_lat", 28.61))
    user_lon = float(d.get("user_lon", 77.21))
    top_k    = int(d.get("top_k", 3))

    results = []
    for _, row in stations_df.iterrows():
        dist = haversine(user_lat, user_lon, float(row["lat"]), float(row["lon"]))
        results.append({
            "station_id":  row["station_id"],
            "lat":         round(float(row["lat"]), 6),
            "lon":         round(float(row["lon"]), 6),
            "distance_km": round(dist, 3),
            "avg_load_kw": round(float(row.get("avg_load", 2000)), 1),
            "avg_price":   round(float(row.get("avg_price", 8)), 2),
            "availability": random.choice(["Available","Busy","Available","Available"])
        })

    results.sort(key=lambda x: x["distance_km"])
    return jsonify({
        "user_location": {"lat": user_lat, "lon": user_lon},
        "nearest_stations": results[:top_k]
    })


# ── 7. Wireless Efficiency ───────────────────────────────────────────────────
@app.route("/wireless_efficiency", methods=["POST"])
def wireless_efficiency():
    d = request.get_json(force=True)
    feat = models.get("wireless_efficiency", {}).get("features", [])
    pred, err = safe_predict("wireless_efficiency", feat, d)
    if err:
        return jsonify({"error": err}), 500

    eff = float(pred[0])
    return jsonify({
        "wireless_efficiency_pct": round(eff, 2),
        "air_gap_cm": d.get("air_gap_cm", 0),
        "recommendation": (
            "Optimal alignment." if eff > 88 else
            "Adjust vehicle position for better coil alignment."
        )
    })


# ── 8. Live Real-time Data ───────────────────────────────────────────────────
@app.route("/live_data", methods=["GET"])
def live_data():
    hour = datetime.now().hour
    is_peak = (7 <= hour <= 9) or (17 <= hour <= 20)
    base_load = 5000 if is_peak else 2500
    base_price = 12.0 if is_peak else 6.5

    return jsonify({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_load_kw": round(base_load + random.uniform(-300, 300), 1),
        "grid_demand_kw":  round(base_load * 1.6 + random.uniform(-400, 400), 1),
        "price_per_kwh":   round(base_price + random.uniform(-1, 1), 2),
        "active_sessions": random.randint(3, 15),
        "stations_available": random.randint(2, 8),
        "avg_battery_health": round(random.uniform(72, 91), 1),
        "thermal_events":  random.randint(0, 2),
        "v2g_energy_kw":   round(random.uniform(0, 50), 1),
        "carbon_saved_kg": round(random.uniform(0.5, 5.0), 2),
        "grid_trend": [
            round(base_load + random.uniform(-500, 500), 1) for _ in range(12)
        ],
        "price_trend": [
            round(base_price + random.uniform(-2, 2), 2) for _ in range(12)
        ]
    })


# ── 9. All Stations ──────────────────────────────────────────────────────────
@app.route("/stations", methods=["GET"])
def get_stations():
    data = []
    for _, row in stations_df.iterrows():
        data.append({
            "station_id": row["station_id"],
            "lat": round(float(row["lat"]), 6),
            "lon": round(float(row["lon"]), 6),
            "avg_price": round(float(row.get("avg_price", 8)), 2),
            "avg_load_kw": round(float(row.get("avg_load", 2000)), 1),
            "sessions": int(row.get("sessions", 0)),
            "status": random.choice(["Online","Online","Online","Busy","Maintenance"])
        })
    return jsonify({"stations": data, "total": len(data)})


# ── 10. AI Chatbot ───────────────────────────────────────────────────────────
CHATBOT_KB = {
    "price": "Current charging prices depend on time-of-day. Off-peak (11 PM–5 AM) rates are cheapest at ₹4–6/kWh. Peak hours (7–9 AM, 5–8 PM) can reach ₹12–18/kWh.",
    "battery": "Battery health depends on charging cycles, temperature, and voltage. Avoid charging above 90% regularly and keep temperatures below 40°C for best longevity.",
    "thermal": "Thermal runaway is a dangerous chain reaction where heat causes more heat. Our AI monitors temperature and voltage in real-time and auto-cuts power if risk exceeds 75%.",
    "station": "Use the map or /nearest_station API to find the closest charging station. We have 10 stations across the metro area.",
    "wireless": "Wireless charging efficiency is 85–95% depending on coil alignment and air gap. Keep the air gap under 5 cm for optimal performance.",
    "v2g": "Vehicle-to-Grid (V2G) lets your EV feed energy back to the grid during peak demand, earning you credits. It activates when battery > 80% and grid demand > 8000 kW.",
    "carbon": "Your EV charging sessions generate ~0.82 kg CO₂/kWh based on the Indian grid. V2G participation and off-peak charging reduce your carbon footprint.",
    "hello": "Hello! I'm your EV Assistant. Ask me about charging prices, battery health, nearest stations, or safety tips!",
    "hi": "Hi there! How can I help with your EV charging today?",
    "default": "I can help with: charging prices, battery health, thermal safety, nearest stations, wireless charging, V2G, and carbon tracking. What would you like to know?"
}

@app.route("/chatbot", methods=["POST"])
def chatbot():
    d = request.get_json(force=True)
    msg = d.get("message", "").lower()
    response = CHATBOT_KB["default"]
    for kw, ans in CHATBOT_KB.items():
        if kw in msg:
            response = ans
            break
    return jsonify({"response": response, "timestamp": datetime.now().strftime("%H:%M")})


# ── 11. Carbon Stats ─────────────────────────────────────────────────────────
@app.route("/carbon_stats", methods=["GET"])
def carbon_stats():
    return jsonify({
        "today_emissions_kg": round(random.uniform(12, 45), 2),
        "monthly_emissions_kg": round(random.uniform(300, 900), 2),
        "v2g_saved_kg": round(random.uniform(5, 30), 2),
        "equivalent_trees": round(random.uniform(1, 8), 1),
        "grid_emission_factor": 0.82,
        "tip": "Charging at night (11 PM–5 AM) reduces carbon footprint by up to 35% due to cleaner grid energy."
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EV Charging Station API — Starting Flask Server")
    print("  URL: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
