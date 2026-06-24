"""
train_model.py
==============
Pipeline pelatihan model Machine Learning untuk prediksi
Heating Load dan Cooling Load pada bangunan.

Mengikuti metodologi CRISP-DM:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Interpretation & Recommendation

Author  : Kelompok Energi
Dataset : UCI Energy Efficiency Dataset (768 samples)
Model   : MultiOutputRegressor(RandomForestRegressor)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
MODEL_PATH   = "model/energy_model.pkl"
SCALER_PATH  = "model/scaler.pkl"
FEATURE_PATH = "model/feature_names.pkl"

FEATURE_NAMES = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution",
]
TARGET_NAMES = ["Heating Load", "Cooling Load"]

os.makedirs("model", exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset():
    """Load dataset dari URL UCI atau file lokal."""
    print("\n" + "=" * 55)
    print("  CRISP-DM: DATA UNDERSTANDING")
    print("=" * 55)

    try:
        print("  Mengunduh dataset dari UCI Repository...")
        df = pd.read_excel(DATASET_URL, header=0)
    except Exception:
        print("  [WARN] Gagal mengunduh. Membuat dataset sintetis...")
        df = _generate_synthetic_dataset()

    # Ambil 10 kolom pertama yang relevan saja
    df = df.iloc[:, :10]
    df.columns = FEATURE_NAMES + TARGET_NAMES

    print(f"  ✔ Dataset berhasil dimuat: {df.shape[0]} baris × {df.shape[1]} kolom")
    print(f"\n  Ringkasan statistik:")
    print(df.describe().to_string())
    return df


def _generate_synthetic_dataset():
    """Membuat dataset sintetis menyerupai UCI Energy Efficiency."""
    np.random.seed(42)
    n = 768
    RC  = np.random.uniform(0.62, 0.98, n)       # Relative Compactness
    SA  = np.random.uniform(514, 808, n)          # Surface Area
    WA  = np.random.uniform(245, 416, n)          # Wall Area
    RA  = np.random.uniform(110, 221, n)          # Roof Area
    OH  = np.random.choice([3.5, 7.0], n)         # Overall Height
    OR  = np.random.choice([2, 3, 4, 5], n)       # Orientation
    GA  = np.random.choice([0, 0.1, 0.25, 0.4], n)   # Glazing Area
    GAD = np.random.choice([0, 1, 2, 3, 4, 5], n)    # Glazing Distribution

    HL = (25 - 20 * RC + 0.01 * SA + 0.02 * WA
          - 0.05 * RA + 5 * OH + 15 * GA
          + np.random.normal(0, 1.5, n))
    CL = (20 - 15 * RC + 0.008 * SA + 0.015 * WA
          - 0.03 * RA + 4 * OH + 18 * GA
          + np.random.normal(0, 1.5, n))

    return pd.DataFrame({
        "RC": RC, "SA": SA, "WA": WA, "RA": RA, "OH": OH,
        "OR": OR, "GA": GA, "GAD": GAD, "HL": HL, "CL": CL,
    })


# ─────────────────────────────────────────────
# 2. DATA VALIDATION
# ─────────────────────────────────────────────
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  VALIDASI DATA")
    print("=" * 55)

    before = len(df)
    df = df.dropna()
    df = df[df[TARGET_NAMES[0]] > 0]
    df = df[df[TARGET_NAMES[1]] > 0]
    after = len(df)

    print(f"  Baris sebelum validasi : {before}")
    print(f"  Baris setelah validasi : {after}")
    print(f"  Nilai kosong           : {before - after} baris dihapus")
    print(f"  Duplikat               : {df.duplicated().sum()} baris")
    return df


# ─────────────────────────────────────────────
# 3. EDA & CORRELATION HEATMAP
# ─────────────────────────────────────────────
def exploratory_data_analysis(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  CRISP-DM: DATA UNDERSTANDING — EDA")
    print("=" * 55)

    print("\n  Distribusi target:")
    for t in TARGET_NAMES:
        print(f"  {t:20s} → mean={df[t].mean():.2f}  "
              f"std={df[t].std():.2f}  "
              f"min={df[t].min():.2f}  max={df[t].max():.2f}")

    os.makedirs("outputs", exist_ok=True)

    # Heatmap
    plt.figure(figsize=(11, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap — Energy Efficiency Dataset", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150)
    plt.close()
    print("  ✔ Heatmap disimpan → outputs/correlation_heatmap.png")


# ─────────────────────────────────────────────
# 4. DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  CRISP-DM: DATA PREPARATION")
    print("=" * 55)

    X = df[FEATURE_NAMES].values
    y = df[TARGET_NAMES].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"  Split 70:30 → Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print("  ✔ StandardScaler terpasang")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 5. MODELING
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    print("\n" + "=" * 55)
    print("  CRISP-DM: MODELING")
    print("=" * 55)
    print("  Algoritma : MultiOutputRegressor(RandomForestRegressor)")
    print("  n_estimators=200, max_depth=None, random_state=42")
    print("  Melatih model...")

    rf   = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf)
    model.fit(X_train, y_train)

    print("  ✔ Pelatihan selesai")
    return model


# ─────────────────────────────────────────────
# 6. EVALUASI
# ─────────────────────────────────────────────
def mape(y_true, y_pred, eps=1e-9):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 55)
    print("  CRISP-DM: EVALUATION")
    print("=" * 55)

    y_pred = model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred)
    results = {}

    for i, name in enumerate(TARGET_NAMES):
        yt, yp = y_test[:, i], y_pred[:, i]
        m   = mape(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        results[name] = {"MAPE": m, "RMSE": rmse, "R2": r2}

        print(f"\n  [{name}]")
        print(f"    MAPE : {m:.2f}%")
        print(f"    RMSE : {rmse:.4f}")
        print(f"    R²   : {r2:.4f}")

    _plot_feature_importance(model)
    return results

def plot_actual_vs_predicted(y_test, y_pred):
    os.makedirs("outputs", exist_ok=True)

    for i, target in enumerate(TARGET_NAMES):

        plt.figure(figsize=(6,6))

        plt.scatter(
            y_test[:, i],
            y_pred[:, i],
            alpha=0.7
        )

        min_val = min(
            y_test[:, i].min(),
            y_pred[:, i].min()
        )

        max_val = max(
            y_test[:, i].max(),
            y_pred[:, i].max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            linewidth=2
        )

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted - {target}")

        plt.tight_layout()

        plt.savefig(
            f"outputs/actual_vs_predicted_{target.replace(' ','_')}.png",
            dpi=300
        )

        plt.close()

    print("✔ Actual vs Predicted plots disimpan")

def _plot_feature_importance(model):
    importances = np.mean(
        [est.feature_importances_ for est in model.estimators_], axis=0
    )
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        [FEATURE_NAMES[i] for i in idx],
        importances[idx] * 100,
        color=plt.cm.RdYlGn(importances[idx] / importances[idx].max()),
        edgecolor="white",
    )
    plt.title("Feature Importance — Random Forest", fontsize=13)
    plt.ylabel("Importance (%)")
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, importances[idx] * 100):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("\n  ✔ Feature Importance disimpan → outputs/feature_importance.png")


# ─────────────────────────────────────────────
# 7. SIMPAN MODEL
# ─────────────────────────────────────────────
def save_artifacts(model, scaler):
    print("\n" + "=" * 55)
    print("  MENYIMPAN ARTEFAK MODEL")
    print("=" * 55)

    joblib.dump(model,        MODEL_PATH)
    joblib.dump(scaler,       SCALER_PATH)
    joblib.dump(FEATURE_NAMES, FEATURE_PATH)

    print(f"  ✔ Model   → {MODEL_PATH}")
    print(f"  ✔ Scaler  → {SCALER_PATH}")
    print(f"  ✔ Features → {FEATURE_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "█" * 55)
    print("  ENERGY EFFICIENCY PREDICTION — TRAINING PIPELINE")
    print("█" * 55)

    df                             = load_dataset()
    df                             = validate_data(df)
    exploratory_data_analysis(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model                          = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler)

    print("\n" + "█" * 55)
    print("  TRAINING SELESAI — Jalankan: streamlit run app.py")
    print("█" * 55 + "\n")


if __name__ == "__main__":
    main()