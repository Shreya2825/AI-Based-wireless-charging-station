"""
=============================================================================
AI-Based Smart EV Charging Station — Model Training Pipeline
=============================================================================


Models trained:
  1. Battery Health Classifier     (XGBoost / Random Forest)
  2. Load Predictor                (XGBoost Regressor)
  3. Price Predictor               (XGBoost Regressor)
  4. Priority Classifier           (XGBoost)
  5. Overheating Detector          (Random Forest — Safety Module)
  6. Thermal Risk Regressor        (Gradient Boosting)
  7. LSTM Grid Demand Forecaster   (Keras / TensorFlow)
  8. Wireless Efficiency Model     (Linear / Ridge)

Run:
    python model_training.py

Output:
    saved_models.pkl   (all sklearn models)
    lstm_model.h5      (LSTM model — separate due to Keras format)
    training_report.txt
=============================================================================
"""

import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt


if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# ── sklearn ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model    import Ridge
from sklearn.metrics         import (
    accuracy_score, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, confusion_matrix, roc_auc_score
)
from sklearn.pipeline        import Pipeline
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not installed – falling back to RandomForest/GBR")

# ── TensorFlow / Keras ────────────────────────────────────────────────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not installed – LSTM training will be skipped")

DATASET_FILE  = "ev_dataset.csv"
MODELS_FILE   = "saved_models.pkl"
LSTM_FILE     = "lstm_model.h5"
REPORT_FILE   = "training_report.txt"
SEQUENCE_LEN  = 24   # 24 time-steps for LSTM (hourly look-back)

report_lines  = []


def log(msg: str):
    """Print and store to report."""
    print(msg)
    report_lines.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATASET_FILE):
        log(f"[ERROR] Dataset not found: {DATASET_FILE}")
        log("[INFO] Running dataset_generator.py first …")
        os.system(f"{sys.executable} dataset_generator.py")
    df = pd.read_csv(DATASET_FILE)
    log(f"[DATA] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Encode station_id
    le = LabelEncoder()
    df["station_id_enc"] = le.fit_transform(df["station_id"])

    # Encode priority
    priority_map = {"Low": 0, "Medium": 1, "High": 2}
    df["priority_enc"] = df["priority"].map(priority_map)

    # Cyclic encoding for hour & day
    df["hour_sin"] = np.sin(2 * np.pi * df["time_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Derived features
    df["state_of_charge_delta"] = 100 - df["battery_level"]
    df["load_ratio"]            = df["current_load"] / (df["grid_demand"] + 1)
    df["temp_voltage_interaction"] = df["temperature"] * df["voltage"]
    df["health_cycle_ratio"]    = df["battery_health"] / (df["charging_cycles"] + 1)

    log(f"[FEAT] Feature engineering complete. Shape: {df.shape}")
    return df, le


# ─────────────────────────────────────────────────────────────────────────────
# 3. BATTERY HEALTH CLASSIFIER (binary: Good/Poor)
# ─────────────────────────────────────────────────────────────────────────────
def train_battery_health_classifier(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 1 — Battery Health Classifier (XGBoost / RF)")
    log("="*60)

    # Label: health >= 70 → Good (1), else Poor (0)
    y = (df["battery_health"] >= 70).astype(int)
    features = ["battery_level", "charging_cycles", "temperature",
                "voltage", "energy_consumed", "thermal_risk_score",
                "hour_sin", "hour_cos", "health_cycle_ratio", "temp_voltage_interaction"]
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if XGBOOST_AVAILABLE:
        clf = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    log(f"  Accuracy : {acc*100:.2f}%")
    log(f"  ROC-AUC  : {auc:.4f}")
    log(f"\n{classification_report(y_test, y_pred, target_names=['Poor','Good'])}")
    return clf, features, acc


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD PREDICTOR (Regression)
# ─────────────────────────────────────────────────────────────────────────────
def train_load_predictor(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 2 — Load Predictor (XGBoost Regressor)")
    log("="*60)

    features = ["time_of_day", "day_of_week", "month", "battery_level",
                "grid_demand", "station_id_enc", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "energy_consumed", "charging_time"]
    X = df[features]
    y = df["current_load"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if XGBOOST_AVAILABLE:
        reg = XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1)
    else:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    log(f"  MAE  : {mae:.3f} kW")
    log(f"  RMSE : {rmse:.3f} kW")
    log(f"  R²   : {r2:.4f}  ({r2*100:.2f}%)")
    return reg, features, r2


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRICE PREDICTOR (Regression)
# ─────────────────────────────────────────────────────────────────────────────
def train_price_predictor(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 3 — Dynamic Price Predictor (XGBoost Regressor)")
    log("="*60)

    features = ["time_of_day", "day_of_week", "month", "grid_demand",
                "current_load", "battery_health", "load_ratio",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos", "station_id_enc"]
    X = df[features]
    y = df["price_per_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if XGBOOST_AVAILABLE:
        reg = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04,
                           subsample=0.85, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1)
    else:
        reg = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    log(f"  MAE  : ₹{mae:.4f}/kWh")
    log(f"  R²   : {r2:.4f}  ({r2*100:.2f}%)")
    return reg, features, r2


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRIORITY CLASSIFIER (Multi-class: Low / Medium / High)
# ─────────────────────────────────────────────────────────────────────────────
def train_priority_classifier(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 4 — Priority Classifier (XGBoost Multi-class)")
    log("="*60)

    features = ["battery_level", "battery_health", "charging_cycles",
                "temperature", "current_load", "grid_demand",
                "time_of_day", "day_of_week", "thermal_risk_score",
                "hour_sin", "hour_cos", "state_of_charge_delta"]
    X = df[features]
    y = df["priority_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if XGBOOST_AVAILABLE:
        clf = XGBClassifier(
            n_estimators=350, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            num_class=3, objective="multi:softprob",
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        )
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    log(f"  Accuracy : {acc*100:.2f}%")
    log(f"\n{classification_report(y_test, y_pred, target_names=['Low','Medium','High'])}")
    return clf, features, acc


# ─────────────────────────────────────────────────────────────────────────────
# 7. OVERHEATING DETECTOR — Safety Module
# ─────────────────────────────────────────────────────────────────────────────
def train_overheating_detector(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 5 — Overheating Detector / Safety Module (RF)")
    log("="*60)

    features = ["temperature", "voltage", "battery_health",
                "charging_cycles", "current_load", "energy_consumed",
                "charging_time", "thermal_risk_score", "temp_voltage_interaction"]
    X = df[features]
    y = df["is_overheating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: cw[0], 1: cw[1]}

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=10, class_weight=class_weight_dict,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    except Exception:
        auc = 0.0
    log(f"  Accuracy : {acc*100:.2f}%")
    log(f"  ROC-AUC  : {auc:.4f}")
    log(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Overheating'])}")
    return clf, features, acc


# ─────────────────────────────────────────────────────────────────────────────
# 8. THERMAL RISK REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────
def train_thermal_risk_regressor(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 6 — Thermal Risk Score Regressor (GBR)")
    log("="*60)

    features = ["temperature", "voltage", "battery_health",
                "charging_cycles", "current_load", "energy_consumed",
                "temp_voltage_interaction", "battery_level"]
    X = df[features]
    y = df["thermal_risk_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    log(f"  R²   : {r2:.4f}  ({r2*100:.2f}%)")
    log(f"  MAE  : {mae:.3f}")
    return reg, features, r2


# ─────────────────────────────────────────────────────────────────────────────
# 9. LSTM GRID DEMAND FORECASTER
# ─────────────────────────────────────────────────────────────────────────────
def train_lstm_forecaster(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 7 — LSTM Grid Demand Forecaster (TensorFlow/Keras)")
    log("="*60)

    if not TF_AVAILABLE:
        log("[SKIP] TensorFlow not available — skipping LSTM training")
        return None, None, 0.0

    # Use grid_demand sorted by time for sequence modeling
    series = df.sort_values("time_of_day")["grid_demand"].values.astype("float32")

    # Normalise
    scaler_lstm = MinMaxScaler()
    series_scaled = scaler_lstm.fit_transform(series.reshape(-1, 1)).flatten()

    # Build sequences
    def make_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    X_seq, y_seq = make_sequences(series_scaled, SEQUENCE_LEN)
    X_seq = X_seq.reshape(X_seq.shape[0], SEQUENCE_LEN, 1)

    split = int(len(X_seq) * 0.8)
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    # LSTM Architecture
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQUENCE_LEN, 1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    log(f"  LSTM architecture:\n{model.summary()}")

    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5)
    ]
    history = model.fit(
        X_tr, y_tr, epochs=60, batch_size=256,
        validation_split=0.1, callbacks=callbacks, verbose=1
    )

    y_pred_scaled = model.predict(X_te).flatten()
    y_pred  = scaler_lstm.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true  = scaler_lstm.inverse_transform(y_te.reshape(-1, 1)).flatten()

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    log(f"  LSTM R²  : {r2:.4f}  ({r2*100:.2f}%)")
    log(f"  LSTM MAE : {mae:.2f} kW")

    # Save LSTM model and scaler
    model.save(LSTM_FILE)
    joblib.dump(scaler_lstm, "lstm_scaler.pkl")
    log(f"  LSTM model saved: {LSTM_FILE}")

    # Save actual vs predicted for plotting
    pd.DataFrame({"actual": y_true[:500], "predicted": y_pred[:500]}).to_csv(
        "lstm_predictions.csv", index=False
    )

    return model, scaler_lstm, r2


# ─────────────────────────────────────────────────────────────────────────────
# 10. WIRELESS EFFICIENCY MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_wireless_efficiency_model(df: pd.DataFrame):
    log("\n" + "="*60)
    log("MODEL 8 — Wireless Charging Efficiency (Ridge Regression)")
    log("="*60)

    features = ["air_gap_cm", "temperature", "voltage", "current_load",
                "battery_level", "energy_consumed"]
    X = df[features]
    y = df["wireless_efficiency_pct"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    reg = Pipeline([("scaler", scaler), ("ridge", Ridge(alpha=1.0))])
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    log(f"  R²   : {r2:.4f}")
    log(f"  MAE  : {mae:.3f}%")
    return reg, features, r2


# ─────────────────────────────────────────────────────────────────────────────
# 11. SAVE ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
def save_models(models_dict: dict):
    joblib.dump(models_dict, MODELS_FILE)
    size_mb = os.path.getsize(MODELS_FILE) / (1024 * 1024)
    log(f"\n[SAVE] All models saved to '{MODELS_FILE}' ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE STATION INFO (for nearest_station API)
# ─────────────────────────────────────────────────────────────────────────────
def save_station_info(df: pd.DataFrame):
    stations_agg = df.groupby("station_id").agg(
        lat=("station_lat", "mean"),
        lon=("station_lon", "mean"),
        avg_load=("current_load", "mean"),
        avg_price=("price_per_kwh", "mean"),
        sessions=("battery_level", "count")
    ).reset_index()
    stations_agg.to_csv("stations_info.csv", index=False)
    log(f"[SAVE] Station info saved to 'stations_info.csv'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=" * 70)
    log("  AI-BASED SMART EV CHARGING STATION — MODEL TRAINING PIPELINE")
    log("=" * 70)

    # 1. Load & engineer features
    df = load_data()
    df, label_encoder = engineer_features(df)

    # 2. Train all models
    clf_health,   feat_health,    acc_health    = train_battery_health_classifier(df)
    reg_load,     feat_load,      r2_load       = train_load_predictor(df)
    reg_price,    feat_price,     r2_price      = train_price_predictor(df)
    clf_priority, feat_priority,  acc_priority  = train_priority_classifier(df)
    clf_overheat, feat_overheat,  acc_overheat  = train_overheating_detector(df)
    reg_thermal,  feat_thermal,   r2_thermal    = train_thermal_risk_regressor(df)
    _,            _,              r2_lstm       = train_lstm_forecaster(df)
    reg_wireless, feat_wireless,  r2_wireless   = train_wireless_efficiency_model(df)

    # 3. Package and save
    save_station_info(df)
    all_models = {
        "battery_health_classifier": {
            "model": clf_health, "features": feat_health, "accuracy": acc_health
        },
        "load_predictor": {
            "model": reg_load, "features": feat_load, "r2": r2_load
        },
        "price_predictor": {
            "model": reg_price, "features": feat_price, "r2": r2_price
        },
        "priority_classifier": {
            "model": clf_priority, "features": feat_priority, "accuracy": acc_priority
        },
        "overheating_detector": {
            "model": clf_overheat, "features": feat_overheat, "accuracy": acc_overheat
        },
        "thermal_risk_regressor": {
            "model": reg_thermal, "features": feat_thermal, "r2": r2_thermal
        },
        "wireless_efficiency": {
            "model": reg_wireless, "features": feat_wireless, "r2": r2_wireless
        },
        "label_encoder": label_encoder,
    }
    save_models(all_models)

    # 4. Summary
    elapsed = time.time() - t0
    log("\n" + "=" * 70)
    log("  TRAINING SUMMARY")
    log("=" * 70)
    log(f"  Battery Health Accuracy  : {acc_health*100:.2f}%")
    log(f"  Load Prediction R²       : {r2_load*100:.2f}%")
    log(f"  Price Prediction R²      : {r2_price*100:.2f}%")
    log(f"  Priority Accuracy        : {acc_priority*100:.2f}%")
    log(f"  Overheating Accuracy     : {acc_overheat*100:.2f}%")
    log(f"  Thermal Risk R²          : {r2_thermal*100:.2f}%")
    log(f"  LSTM Demand R²           : {r2_lstm*100:.2f}%" if r2_lstm else "  LSTM: Skipped")
    log(f"  Wireless Efficiency R²   : {r2_wireless*100:.2f}%")
    log(f"\n  Total training time      : {elapsed:.1f}s")
    log("=" * 70)

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))
    log(f"[DONE] Report saved to '{REPORT_FILE}'")


if __name__ == "__main__":
    main()
