"""
helper.py
=========
Fungsi-fungsi pembantu untuk aplikasi Streamlit prediksi
Heating Load & Cooling Load bangunan.

Berisi:
  - load_model()            : memuat model, scaler, feature names
  - validate_inputs()       : validasi form input pengguna
  - predict()               : jalankan prediksi multi-output
  - energy_efficiency_score(): skor 0-100 dan kategori efisiensi
  - get_feature_importance(): persentase kontribusi fitur
  - get_recommendation()    : rekomendasi berbasis fitur dominan
  - make_pmap_figure()      : Performance Map (P-Map) Plotly
  - make_importance_figure(): Bar chart Feature Importance Plotly
  - what_if_analysis()      : simulasi perubahan satu fitur
  - estimate_cost()         : estimasi biaya operasional bulanan
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import shap

# ──────────────────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────────────────
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

# Tarif PLN default (Rp/kWh) — bisa diubah user
DEFAULT_PLN_RATE = 1_444   # Rp/kWh (golongan R-2/R-3, 2024)
BUILDING_AREA_M2 = 100     # asumsi luas bangunan default


# ──────────────────────────────────────────────────────────
# 1. LOAD MODEL
# ──────────────────────────────────────────────────────────
def load_model():
    """Memuat model, scaler, dan nama fitur dari disk."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURE_PATH]):
        return None, None, None

    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, scaler, features


# ──────────────────────────────────────────────────────────
# 2. VALIDASI INPUT
# ──────────────────────────────────────────────────────────
def validate_inputs(inputs: dict) -> tuple[bool, list[str]]:
    """
    Validasi dictionary input bangunan.
    Mengembalikan (is_valid: bool, errors: list[str]).
    """
    errors = []

    rc = inputs.get("Relative Compactness", -1)
    if not (0.0 < rc <= 1.0):
        errors.append("Relative Compactness harus antara 0 (eksklusif) dan 1.")

    area_fields = ["Surface Area", "Wall Area", "Roof Area"]
    for field in area_fields:
        val = inputs.get(field, 0)
        if val <= 0:
            errors.append(f"{field} harus lebih dari 0.")

    for key, val in inputs.items():
        if val < 0:
            errors.append(f"{key} tidak boleh bernilai negatif.")

    ga = inputs.get("Glazing Area", -1)
    if not (0.0 <= ga <= 1.0):
        errors.append("Glazing Area harus antara 0 dan 1.")

    return (len(errors) == 0), errors


# ──────────────────────────────────────────────────────────
# 3. PREDIKSI
# ──────────────────────────────────────────────────────────
def predict(model, scaler, inputs: dict) -> dict:
    """
    Menjalankan prediksi Heating Load dan Cooling Load.
    inputs : dict dengan key = FEATURE_NAMES
    returns: dict {"Heating Load": float, "Cooling Load": float}
    """
    x = np.array([[inputs[f] for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return {
        "Heating Load": round(float(pred[0]), 6),
        "Cooling Load": round(float(pred[1]), 6),
    }


# ──────────────────────────────────────────────────────────
# 4. ENERGY EFFICIENCY SCORE
# ──────────────────────────────────────────────────────────
def energy_efficiency_score(hl: float, cl: float) -> tuple[int, str, str]:
    """
    Menghitung Energy Efficiency Score (0-100) dan kategori.

    Pendekatan:
      - Rata-rata HL dan CL dari dataset UCI ≈ 22 kWh/m²
      - Nilai maksimum efisiensi referensi = 10 kWh/m² (sangat hemat)
      - Nilai minimum = 45 kWh/m² (sangat boros)

    Returns: (score: int, category: str, color: str)
    """
    avg = (hl + cl) / 2
    # Normalisasi: skor 100 jika avg ≤ 10, skor 0 jika avg ≥ 45
    score = int(np.clip((45 - avg) / (45 - 10) * 100, 0, 100))

    if score >= 80:
        return score, "Tinggi", "#22c55e"
    elif score >= 60:
        return score, "Sedang", "#f59e0b"
    else:
        return score, "Rendah", "#ef4444"


# ──────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────
def get_feature_importance(model) -> pd.DataFrame:
    """
    Mengambil rata-rata feature importance dari semua estimator.
    Returns DataFrame dengan kolom: Feature, Importance, Pct
    """
    importances = np.mean(
        [est.feature_importances_ for est in model.estimators_], axis=0
    )
    total = importances.sum()
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
        "Pct": (importances / total * 100).round(1),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df

def get_local_explanation(model, scaler, inputs):
    """
    Feature importance yang menyesuaikan dengan input user
    """

    x = np.array([[inputs[f] for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    # pakai estimator pertama (Heating Load)
    explainer = shap.TreeExplainer(model.estimators_[0])

    shap_values = explainer.shap_values(x_scaled)

    contribution = np.abs(shap_values[0])

    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Contribution": contribution
    })

    df["Pct"] = (
        df["Contribution"]
        / df["Contribution"].sum()
        * 100
    ).round(1)

    df = df.sort_values(
        "Pct",
        ascending=False
    ).reset_index(drop=True)

    return df

# ──────────────────────────────────────────────────────────
# 6. REKOMENDASI OTOMATIS
# ──────────────────────────────────────────────────────────
RECOMMENDATION_MAP = {
    "Relative Compactness": (
        "🏠 Kompaktkan Desain Bangunan\n\n"
        "Relative Compactness merupakan faktor dominan pada prediksi ini. "
        "Semakin kompak sebuah bangunan, semakin kecil permukaan yang bersentuhan "
        "dengan lingkungan luar, sehingga kehilangan energi berkurang.\n\n"
        "→ Pertimbangkan desain yang lebih kompak dan hindari bentuk bangunan "
        "yang terlalu memanjang atau berliku."
    ),
    "Surface Area": (
        "📐 Optimasi Luas Permukaan\n\n"
        "Luas permukaan yang besar memperbesar area pertukaran panas antara "
        "bangunan dan lingkungan. Hal ini meningkatkan kebutuhan pendinginan "
        "maupun pemanasan.\n\n"
        "→ Kurangi luas permukaan total atau tingkatkan kualitas insulasi pada "
        "seluruh permukaan eksterior bangunan."
    ),
    "Wall Area": (
        "🧱 Tingkatkan Insulasi Dinding\n\n"
        "Wall Area adalah faktor yang paling memengaruhi prediksi ini. "
        "Dinding dengan luas besar atau insulasi rendah meningkatkan transfer "
        "panas secara signifikan.\n\n"
        "→ Gunakan material dinding dengan nilai insulasi termal (R-value) "
        "yang lebih tinggi, seperti bata berongga, panel sandwich, atau "
        "dinding berlapis insulasi mineral."
    ),
    "Roof Area": (
        "🏚️ Optimalkan Desain Atap\n\n"
        "Atap adalah area yang menerima paparan radiasi matahari terbesar. "
        "Roof Area besar tanpa insulasi yang memadai meningkatkan Cooling Load "
        "secara drastis.\n\n"
        "→ Pertimbangkan pengurangan luas atap, penambahan insulasi atap, "
        "penggunaan cool roof coating, atau pemasangan panel surya untuk "
        "mengurangi panas yang masuk."
    ),
    "Overall Height": (
        "🏢 Evaluasi Ketinggian Bangunan\n\n"
        "Ketinggian bangunan berpengaruh pada volume udara yang harus "
        "dikondisikan serta paparan angin.\n\n"
        "→ Pertimbangkan ceiling height yang optimal dan pastikan sistem "
        "HVAC dimensinya sesuai dengan volume bangunan."
    ),
    "Orientation": (
        "🧭 Optimasi Orientasi Bangunan\n\n"
        "Orientasi bangunan menentukan seberapa besar paparan sinar matahari "
        "langsung yang diterima fasad. Orientasi yang salah dapat "
        "meningkatkan Cooling Load secara signifikan.\n\n"
        "→ Arahkan sisi bangunan yang memiliki kaca/jendela terbesar menghadap "
        "Utara atau Selatan untuk mengurangi paparan matahari pagi dan sore."
    ),
    "Glazing Area": (
        "🪟 Kurangi atau Tingkatkan Kualitas Kaca\n\n"
        "Glazing Area (luas kaca) adalah faktor TERBESAR pada prediksi ini. "
        "Kaca adalah konduktor panas yang buruk — kaca biasa memiliki nilai "
        "U-value yang sangat tinggi, menyebabkan transfer panas masif.\n\n"
        "→ Pertimbangkan mengurangi luas kaca total, menggunakan **low-e glass** "
        "(emisi rendah), double-glazed, atau triple-glazed untuk mengurangi "
        "transfer panas hingga 50%."
    ),
    "Glazing Area Distribution": (
        "🔲 Distribusikan Kaca Secara Strategis\n\n"
        "Distribusi area kaca per sisi bangunan sangat memengaruhi efisiensi. "
        "Kaca yang terkonsentrasi pada sisi Barat/Timur memaksimalkan "
        "paparan matahari pagi dan sore.\n\n"
        "→ Distribusikan kaca secara merata atau fokuskan pada sisi Utara/Selatan "
        "untuk mengurangi panas masuk (solar heat gain) secara efektif."
    ),
}


def get_recommendation(feature_importance_df: pd.DataFrame) -> list[dict]:
    """
    Mengembalikan daftar rekomendasi berdasarkan urutan Feature Importance.
    Returns: list of dict {"rank", "feature", "pct", "text"}
    """
    recs = []
    for rank, row in feature_importance_df.head(4).iterrows():
        feat = row["Feature"]
        text = RECOMMENDATION_MAP.get(feat, "Perhatikan desain pada aspek ini.")
        recs.append({
            "rank": rank + 1,
            "feature": feat,
            "pct": row["Pct"],
            "text": text,
        })
    return recs


# ──────────────────────────────────────────────────────────
# 7. PERFORMANCE MAP (P-MAP)
# ──────────────────────────────────────────────────────────
def make_pmap_figure(hl_user: float, cl_user: float) -> go.Figure:
    """
    Membuat Performance Map (P-Map) dengan zona efisiensi dan
    posisi bangunan pengguna.
    """
    # Grid zona efisiensi
    hl_range = np.linspace(5, 50, 200)
    cl_range = np.linspace(5, 50, 200)
    HL_grid, CL_grid = np.meshgrid(hl_range, cl_range)
    avg_grid = (HL_grid + CL_grid) / 2

    # Skor zona
    score_grid = np.clip((45 - avg_grid) / (45 - 10) * 100, 0, 100)

    fig = go.Figure()

    # Kontour zona
    fig.add_trace(go.Contour(
        x=hl_range,
        y=cl_range,
        z=score_grid,
        colorscale=[
            [0.00, "#fca5a5"],
            [0.30, "#fcd34d"],
            [0.60, "#86efac"],
            [1.00, "#16a34a"],
        ],
        contours=dict(
            start=0, end=100, size=20,
            showlabels=True,
            labelfont=dict(size=10, color="white"),
        ),
        colorbar=dict(
            title="Efficiency Score",
            tickvals=[0, 20, 40, 60, 80, 100],
            ticktext=["0", "20", "40 Rendah", "60", "80 Sedang", "100 Tinggi"],
        ),
        opacity=0.75,
        name="Zona Efisiensi",
        hovertemplate="HL=%{x:.1f}  CL=%{y:.1f}  Score=%{z:.0f}<extra></extra>",
    ))

    # Dataset referensi (titik sampel acak)
    np.random.seed(0)
    n_ref = 80
    hl_ref = np.random.uniform(6, 45, n_ref)
    cl_ref = np.random.uniform(6, 45, n_ref)
    fig.add_trace(go.Scatter(
        x=hl_ref, y=cl_ref,
        mode="markers",
        marker=dict(size=5, color="rgba(255,255,255,0.35)", line=dict(width=1, color="white")),
        name="Referensi Dataset",
        hovertemplate="HL=%{x:.1f}  CL=%{y:.1f}<extra>Referensi</extra>",
    ))

    # Posisi bangunan pengguna
    score_user, cat, color = energy_efficiency_score(hl_user, cl_user)
    fig.add_trace(go.Scatter(
        x=[hl_user], y=[cl_user],
        mode="markers+text",
        marker=dict(size=18, color=color, symbol="star", line=dict(width=2, color="white")),
        text=[f"  Anda\n({score_user}/100)"],
        textposition="middle right",
        textfont=dict(size=12, color="white"),
        name=f"Bangunan Anda ({cat})",
        hovertemplate=(
            f"<b>Bangunan Anda</b><br>"
            f"Heating Load : {hl_user} kWh/m²<br>"
            f"Cooling Load : {cl_user} kWh/m²<br>"
            f"Score        : {score_user}/100<br>"
            f"Kategori     : {cat}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="🗺️ Performance Map — Posisi Efisiensi Energi Bangunan", font=dict(size=16)),
        xaxis=dict(title="Heating Load (kWh/m²)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Cooling Load (kWh/m²)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,40,0.85)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        height=520,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


# ──────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE CHART
# ──────────────────────────────────────────────────────────
def make_importance_figure(fi_df: pd.DataFrame) -> go.Figure:
    """Bar chart Feature Importance (Plotly)."""
    colors = px.colors.sequential.Viridis_r[:len(fi_df)]
    fig = go.Figure(go.Bar(
        x=fi_df["Feature"],
        y=fi_df["Pct"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in fi_df["Pct"]],
        textposition="outside",
        hovertemplate="%{x}<br>Kontribusi: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="📊 Kontribusi Fitur terhadap Prediksi (Feature Importance)",
        xaxis=dict(title="Fitur Bangunan"),
        yaxis=dict(title="Kontribusi (%)", range=[0, fi_df["Pct"].max() * 1.25]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,40,0.85)",
        font=dict(color="white"),
        height=400,
        margin=dict(l=40, r=40, t=60, b=100),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


# ──────────────────────────────────────────────────────────
# 9. WHAT-IF ANALYSIS
# ──────────────────────────────────────────────────────────
def what_if_analysis(
    model,
    scaler,
    base_inputs: dict,
    feature_to_change: str,
    pct_change: float,
) -> dict:
    """
    Simulasi perubahan satu fitur.

    Parameters
    ----------
    base_inputs      : input asli pengguna
    feature_to_change: nama fitur yang diubah
    pct_change       : perubahan dalam persen (mis. -10 untuk turun 10%)

    Returns dict dengan keys:
        before_hl, before_cl, before_score, before_cat,
        after_hl, after_cl, after_score, after_cat,
        delta_hl, delta_cl, delta_score
    """
    # Prediksi sebelum
    before_pred = predict(model, scaler, base_inputs)
    bhl = before_pred["Heating Load"]
    bcl = before_pred["Cooling Load"]
    bscore, bcat, _ = energy_efficiency_score(bhl, bcl)

    # Modifikasi fitur
    modified = base_inputs.copy()
    original_val = modified[feature_to_change]
    new_val = original_val * (1 + pct_change / 100)

    # Klem nilai agar tetap valid
    if feature_to_change == "Relative Compactness":
        new_val = np.clip(new_val, 0.01, 1.0)
    elif feature_to_change == "Glazing Area":
        new_val = np.clip(new_val, 0.0, 1.0)
    else:
        new_val = max(new_val, 0.01)

    modified[feature_to_change] = new_val

    # Prediksi sesudah
    after_pred = predict(model, scaler, modified)
    ahl = after_pred["Heating Load"]
    acl = after_pred["Cooling Load"]
    ascore, acat, _ = energy_efficiency_score(ahl, acl)

    return {
        "before_hl": bhl, "before_cl": bcl,
        "before_score": bscore, "before_cat": bcat,
        "after_hl": ahl,  "after_cl": acl,
        "after_score": ascore, "after_cat": acat,
        "delta_hl": round(ahl - bhl, 2),
        "delta_cl": round(acl - bcl, 2),
        "delta_score": ascore - bscore,
        "modified_val": round(new_val, 4),
        "original_val": round(original_val, 4),
    }


# ──────────────────────────────────────────────────────────
# 10. ESTIMASI BIAYA OPERASIONAL
# ──────────────────────────────────────────────────────────
def estimate_cost(
    hl: float,
    cl: float,
    area_m2: float = BUILDING_AREA_M2,
    pln_rate: float = DEFAULT_PLN_RATE,
) -> dict:
    """
    Estimasi biaya listrik bulanan.

    Asumsi:
    - Konsumsi = (HL + CL) × area_m2 kWh/bulan
    - Tarif PLN (Rp/kWh) dapat dikustomisasi

    Returns: dict dengan konsumsi_kwh dan biaya_rp per bulan
    """
    total_load_kwh = (hl + cl) * area_m2          # kWh/bulan
    biaya_rp       = total_load_kwh * pln_rate     # Rp/bulan

    return {
        "konsumsi_kwh": round(total_load_kwh, 1),
        "biaya_rp"    : round(biaya_rp, 0),
        "pln_rate"    : pln_rate,
        "area_m2"     : area_m2,
    }


# ──────────────────────────────────────────────────────────
# HELPER: Format Rupiah
# ──────────────────────────────────────────────────────────
def fmt_rp(value: float) -> str:
    """Format angka ke string Rupiah. Mis: 317680 → 'Rp 317.680'"""
    return f"Rp {int(value):,}".replace(",", ".")