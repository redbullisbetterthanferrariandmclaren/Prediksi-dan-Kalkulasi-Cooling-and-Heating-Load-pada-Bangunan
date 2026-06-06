"""
app.py
======
Aplikasi Streamlit — Prediksi & Kalkulasi Cooling and Heating Load Bangunan

Fitur:
  1. Input Desain Bangunan (Sidebar)
  2. Validasi Input
  3. Prediksi (Heating Load & Cooling Load)
  4. Energy Efficiency Score
  5. Performance Map (P-Map)
  6. Explainable AI (Feature Importance)
  7. Rekomendasi Otomatis
  8. What-If Analysis
  9. Estimasi Biaya Operasional

Jalankan:
  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd

from helper import (
    load_model,
    validate_inputs,
    predict,
    energy_efficiency_score,
    get_feature_importance,
    get_recommendation,
    make_pmap_figure,
    make_importance_figure,
    what_if_analysis,
    estimate_cost,
    fmt_rp,
    FEATURE_NAMES,
    TARGET_NAMES,
    DEFAULT_PLN_RATE,
    BUILDING_AREA_M2,
)

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyPredict AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────
st.markdown("""

<style>
    :root {
        color-scheme: dark !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');
          
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(
        135deg,
        #2B1F2F 0%,
        #35263A 50%,
        #402C46 100%
    );
}

  /* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #F3EEF1 0%,
        #ECE3E8 100%
    );
    border-right: 2px solid #EA9CAF;
}

[data-testid="stSidebar"] * {
    color: #5C4050 !important;
}
            
  /* Main header */
  .main-header {
    background: linear-gradient(
        135deg,
        rgba(234,156,175,0.15) 0%,
        rgba(213,105,137,0.15) 100%
    );
    border: 1px solid rgba(234,156,175,0.4);
  }
  .main-header h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #F3EEF1;
    letter-spacing: 1px;
    margin: 0;
  }
  .main-header p { color: #E7DCE2; margin: 0.4rem 0 0; font-size: 0.95rem; }

  /* Metric cards */
  .metric-card {
    background: rgba(58,41,63,0.85);
    border: 1px solid rgba(234,156,175,0.4);
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0;
    text-align: center;
    backdrop-filter: blur(6px);
  }
  .metric-label { font-size: 0.78rem; color: #E7DCE2; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
  .metric-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2rem; font-weight: 700; color: #F3EEF1; }
  .metric-unit  { font-size: 0.8rem; color: #D6B9C3; }

  /* Score badge */
  .score-badge {
    display: inline-block;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    padding: 0.5rem 1.5rem;
    border-radius: 12px;
    margin: 0.5rem 0;
  }

  /* Section header */
  .section-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #EA9CAF;
    border-left: 4px solid #D56989;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
  }

  /* Info box */
  .info-box {
    background: rgba(243,238,241,0.08);
    border: 1px solid rgba(234,156,175,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #F3EEF1;
    font-size: 0.88rem;
    line-height: 1.6;
  }

  /* Recommendation card */
  .rec-card {
    background: linear-gradient(
        135deg,
        rgba(234,156,175,0.10) 0%,
        rgba(213,105,137,0.10) 100%
    );
    border: 1px solid rgba(234,156,175,0.35);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
    color: #F3EEF1;
    line-height: 1.7;
  }

  /* What-if comparison */
  .compare-before {
    background: rgba(213,105,137,0.08);
    border: 1px solid rgba(213,105,137,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .compare-after {
    background: rgba(194,220,128,0.08);
    border: 1px solid rgba(194,220,128,0.35);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
  }

  /* Hide Streamlit default elements */
  #MainMenu, footer { visibility: hidden; }

  /* Tab style */
  .stTabs [data-baseweb="tab"] {
    background: #F3EEF1;
    color: #7D5768;
    border-radius: 12px 12px 0 0;
    margin-right: 6px;
    padding: 10px 18px;
    font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: #EA9CAF !important;
    color: white !important;
}

  /* Divider */
  hr {
    border-color: rgba(234,156,175,0.25) !important;
}

  /* Number input */
  .stNumberInput input {
    background: rgba(243,238,241,0.08) !important;
    color: #F3EEF1 !important;
    border: 1px solid rgba(234,156,175,0.3) !important;
}

/* Selectbox */
.stSelectbox * {
    color: #F3EEF1 !important;
}

  /* Button */
  .stButton > button {
    background: linear-gradient(
        135deg,
        #EA9CAF 0%,
        #D56989 100%
    );
    color: white;
    border: none;
    border-radius: 12px;
}
  .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# LOAD MODEL (cache)
# ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model AI...")
def get_model():
    return load_model()


model, scaler, feature_names = get_model()
model_loaded = model is not None


# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚡ EnergyPredict AI</h1>
  <p>Prediksi & Kalkulasi Heating Load · Cooling Load · Efisiensi Energi Bangunan</p>
  <p style="font-size:0.78rem;margin-top:0.5rem;color:#475569;">
    Berbasis UCI Energy Efficiency Dataset · Random Forest Regressor · MultiOutputRegressor
  </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "⚠️ **Model belum tersedia.** Jalankan `python train_model.py` terlebih dahulu "
        "untuk melatih dan menyimpan model, kemudian refresh halaman ini."
    )
    st.stop()


# ──────────────────────────────────────────────────────────
# SIDEBAR — INPUT DESAIN BANGUNAN
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("Input Desain Bangunan")
    st.markdown("---")

    st.markdown("#### Karakteristik Geometri")
    rc  = st.slider("Relative Compactness",  0.01, 1.00, 0.76, 0.01,
                    help="Rasio volume bangunan terhadap permukaan (0–1)")
    sa  = st.number_input("Surface Area (m²)",    min_value=1.0, max_value=2000.0, value=661.5, step=1.0)
    wa  = st.number_input("Wall Area (m²)",        min_value=1.0, max_value=2000.0, value=318.5, step=1.0)
    ra  = st.number_input("Roof Area (m²)",        min_value=1.0, max_value=2000.0, value=122.5, step=0.5)
    oh  = st.selectbox("Overall Height (m)",    [3.5, 7.0], index=1,
                       help="Pilih tinggi bangunan: 3.5m (1 lantai) atau 7.0m (2 lantai)")

    st.markdown("---")
    st.markdown("#### Orientasi & Kaca")
    orient_labels = {2: "2 — Utara", 3: "3 — Timur", 4: "4 — Selatan", 5: "5 — Barat"}
    orient_sel = st.selectbox("Orientation", list(orient_labels.keys()),
                              format_func=lambda x: orient_labels[x],
                              help="Arah hadap fasad utama bangunan")
    ga  = st.slider("Glazing Area (rasio)", 0.00, 1.00, 0.25, 0.05,
                    help="Proporsi luas kaca terhadap total fasad (0–1)")
    gad_labels = {0: "0 — Tidak ada kaca", 1: "1 — Seragam", 2: "2 — N>E=W>S",
                  3: "3 — E>W=N>S", 4: "4 — S>E=W>N", 5: "5 — S>N=E>W"}
    gad = st.selectbox("Glazing Area Distribution", list(gad_labels.keys()),
                       format_func=lambda x: gad_labels[x],
                       help="Distribusi kaca per sisi bangunan")

    st.markdown("---")
    st.markdown("#### ⚙️ Pengaturan Biaya")
    pln_rate = st.number_input("Tarif PLN (Rp/kWh)", min_value=500, max_value=5000,
                               value=DEFAULT_PLN_RATE, step=50)
    area_m2  = st.number_input("Luas Bangunan (m²)", min_value=10, max_value=10000,
                               value=BUILDING_AREA_M2, step=10)

    st.markdown("---")
    predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True)


# ──────────────────────────────────────────────────────────
# KUMPULKAN INPUT
# ──────────────────────────────────────────────────────────
user_inputs = {
    "Relative Compactness"      : rc,
    "Surface Area"              : sa,
    "Wall Area"                 : wa,
    "Roof Area"                 : ra,
    "Overall Height"            : oh,
    "Orientation"               : float(orient_sel),
    "Glazing Area"              : ga,
    "Glazing Area Distribution" : float(gad),
}


# ──────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "fi_df" not in st.session_state:
    st.session_state.fi_df = None


# ──────────────────────────────────────────────────────────
# TOMBOL PREDIKSI
# ──────────────────────────────────────────────────────────
if predict_btn:
    is_valid, errors = validate_inputs(user_inputs)
    if not is_valid:
        for err in errors:
            st.error(f"❌ {err}")
        st.session_state.result = None
    else:
        with st.spinner("Menjalankan prediksi AI..."):
            result = predict(model, scaler, user_inputs)
            fi_df  = get_feature_importance(model)
            st.session_state.result    = result
            st.session_state.fi_df    = fi_df
            st.session_state.inputs   = user_inputs.copy()
            st.session_state.pln_rate = pln_rate
            st.session_state.area_m2  = area_m2
        st.success("✅ Prediksi berhasil!")


# ──────────────────────────────────────────────────────────
# MAIN CONTENT TABS
# ──────────────────────────────────────────────────────────
tab_pred, tab_xai, tab_whatif, tab_crisp = st.tabs([
    "Hasil Prediksi",
    "Explainable AI",
    "What-If Analysis",
    "CRISP-DM",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — HASIL PREDIKSI
# ══════════════════════════════════════════════════════════
with tab_pred:
    if st.session_state.result is None:
        st.markdown("""
        <div class="info-box">
          👈 Isi parameter desain bangunan pada sidebar kiri, lalu tekan
          <strong>Prediksi Sekarang</strong> untuk melihat hasil analisis energi.
        </div>
        """, unsafe_allow_html=True)

        # Demo preview P-Map kosong
        st.markdown('<div class="section-header">🗺️ Performance Map (Preview)</div>', unsafe_allow_html=True)
        fig_demo = make_pmap_figure(22.0, 25.0)
        st.plotly_chart(fig_demo, use_container_width=True)

    else:
        res    = st.session_state.result
        fi_df  = st.session_state.fi_df
        hl     = res["Heating Load"]
        cl     = res["Cooling Load"]
        score, cat, color_hex = energy_efficiency_score(hl, cl)
        cost   = estimate_cost(hl, cl,
                               st.session_state.area_m2,
                               st.session_state.pln_rate)

        # ── METRIK UTAMA ──────────────────────────────────
        st.markdown('<div class="section-header">⚡ Hasil Prediksi Energi</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">🌡️ Heating Load</div>
              <div class="metric-value">{hl}</div>
              <div class="metric-unit">kWh/m²</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">❄️ Cooling Load</div>
              <div class="metric-value">{cl}</div>
              <div class="metric-unit">kWh/m²</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            score_bg = {"Tinggi": "rgba(34,197,94,0.15)", "Sedang": "rgba(245,158,11,0.15)", "Rendah": "rgba(239,68,68,0.15)"}[cat]
            st.markdown(f"""
            <div class="metric-card" style="background:{score_bg};">
              <div class="metric-label">⚡ Efficiency Score</div>
              <div class="metric-value" style="color:{color_hex};">{score}</div>
              <div class="metric-unit">/ 100 — {cat}</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">💡 Konsumsi/Bulan</div>
              <div class="metric-value" style="font-size:1.6rem;">{cost['konsumsi_kwh']}</div>
              <div class="metric-unit">kWh · {fmt_rp(cost['biaya_rp'])}</div>
            </div>""", unsafe_allow_html=True)

        # ── PENJELASAN SKOR ───────────────────────────────
        st.markdown("---")
        score_desc = {
            "Tinggi": f"🟢 **Bangunan sangat efisien!** Dengan skor **{score}/100**, desain ini termasuk kategori efisiensi **Tinggi**. Kebutuhan energi bangunan ini rendah dan sudah optimal.",
            "Sedang": f"🟡 **Efisiensi bangunan tergolong sedang.** Skor **{score}/100** menunjukkan ada ruang untuk perbaikan. Perhatikan rekomendasi pada tab XAI untuk meningkatkan efisiensi.",
            "Rendah": f"🔴 **Bangunan boros energi!** Skor **{score}/100** termasuk kategori efisiensi **Rendah**. Segera lakukan penyesuaian desain sebelum konstruksi untuk mengurangi biaya operasional jangka panjang.",
        }
        st.info(score_desc[cat])

        # ── PERFORMANCE MAP ───────────────────────────────
        st.markdown('<div class="section-header">🗺️ Performance Map</div>', unsafe_allow_html=True)
        fig_pmap = make_pmap_figure(hl, cl)
        st.plotly_chart(fig_pmap, use_container_width=True)

        # ── ESTIMASI BIAYA ────────────────────────────────
        st.markdown('<div class="section-header">💰 Estimasi Biaya Operasional</div>', unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.metric("Konsumsi per Bulan", f"{cost['konsumsi_kwh']} kWh",
                      help=f"Luas bangunan {cost['area_m2']} m²")
        with bc2:
            st.metric("Estimasi Biaya/Bulan", fmt_rp(cost['biaya_rp']),
                      help=f"Tarif PLN {fmt_rp(cost['pln_rate'])}/kWh")
        with bc3:
            st.metric("Estimasi Biaya/Tahun", fmt_rp(cost['biaya_rp'] * 12))

        st.caption(
            f"*Estimasi berdasarkan asumsi: luas bangunan {cost['area_m2']} m², "
            f"tarif PLN {fmt_rp(cost['pln_rate'])}/kWh. Angka aktual dapat berbeda.*"
        )

        # ── INPUT SUMMARY ─────────────────────────────────
        with st.expander("📋 Ringkasan Parameter Input"):
            inp_df = pd.DataFrame({
                "Fitur": list(st.session_state.inputs.keys()),
                "Nilai": [round(v, 4) for v in st.session_state.inputs.values()],
            })
            st.dataframe(inp_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════
with tab_xai:
    st.markdown('<div class="section-header">🔬 Explainable AI — Feature Importance</div>', unsafe_allow_html=True)

    if st.session_state.fi_df is None:
        st.info("Jalankan prediksi terlebih dahulu untuk melihat analisis Explainable AI.")
    else:
        fi_df = st.session_state.fi_df
        res   = st.session_state.result
        hl    = res["Heating Load"]
        cl    = res["Cooling Load"]
        score, cat, color_hex = energy_efficiency_score(hl, cl)

        # ── PENJELASAN TOP FEATURE ────────────────────────
        top1  = fi_df.iloc[0]
        top1_pct = top1["Pct"]
        st.markdown(f"""
        <div class="info-box">
          🏆 Faktor yang paling memengaruhi prediksi energi bangunan Anda adalah
          <strong>{top1['Feature']}</strong> dengan kontribusi <strong>{top1_pct:.1f}%</strong>.
          Memodifikasi aspek ini akan memberikan dampak terbesar terhadap efisiensi energi.
        </div>
        """, unsafe_allow_html=True)

        # ── BAR CHART ─────────────────────────────────────
        fig_fi = make_importance_figure(fi_df)
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── TABEL RINCI ───────────────────────────────────
        with st.expander("📊 Tabel Rinci Feature Importance"):
            display_fi = fi_df[["Feature", "Pct"]].copy()
            display_fi.columns = ["Fitur Bangunan", "Kontribusi (%)"]
            st.dataframe(display_fi, use_container_width=True, hide_index=True)

        # ── REKOMENDASI OTOMATIS ──────────────────────────
        st.markdown('<div class="section-header">💡 Rekomendasi Desain Otomatis</div>', unsafe_allow_html=True)
        recs = get_recommendation(fi_df)
        for rec in recs:
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣"][rec["rank"] - 1]
            with st.expander(f"{rank_emoji} {rec['feature']} — Kontribusi {rec['pct']:.1f}%", expanded=(rec["rank"] == 1)):
                st.markdown(f'<div class="rec-card">{rec["text"]}</div>', unsafe_allow_html=True)

        # ── PENJELASAN MATEMATIS ──────────────────────────
        st.markdown('<div class="section-header">📐 Penjelasan Analitik</div>', unsafe_allow_html=True)
        top_features_str = ", ".join([f["feature"] for f in recs[:2]])
        avg_load = (hl + cl) / 2
        avg_ref  = 22.0  # rata-rata dataset UCI
        diff_pct = ((avg_load - avg_ref) / avg_ref) * 100

        if diff_pct > 0:
            math_text = (
                f"Bangunan Anda berada pada zona **{cat}** karena kombinasi "
                f"**{top_features_str}** yang tinggi menyebabkan rata-rata beban energi "
                f"sebesar **{avg_load:.1f} kWh/m²**, yaitu **{abs(diff_pct):.1f}% lebih tinggi** "
                f"dibanding rata-rata bangunan efisien pada dataset UCI ({avg_ref} kWh/m²)."
            )
        else:
            math_text = (
                f"Bangunan Anda berada pada zona **{cat}**. Rata-rata beban energi "
                f"sebesar **{avg_load:.1f} kWh/m²**, yaitu **{abs(diff_pct):.1f}% lebih rendah** "
                f"dibanding rata-rata dataset UCI ({avg_ref} kWh/m²). Desain ini sudah efisien!"
            )

        st.markdown(f'<div class="info-box">📌 {math_text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF ANALYSIS
# ══════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown('<div class="section-header">🔀 What-If Analysis</div>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.info("Jalankan prediksi terlebih dahulu untuk menggunakan What-If Analysis.")
    else:
        st.markdown("""
        <div class="info-box">
          🧭 Simulasikan perubahan pada satu parameter desain dan lihat dampaknya terhadap
          Heating Load, Cooling Load, dan Energy Efficiency Score secara langsung.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        wi_col1, wi_col2 = st.columns([2, 1])
        with wi_col1:
            feature_wi = st.selectbox(
                "Pilih Fitur yang Ingin Diubah",
                FEATURE_NAMES,
                index=FEATURE_NAMES.index("Glazing Area"),
            )
        with wi_col2:
            pct_wi = st.slider("Perubahan (%)", -50, 50, -10, 5,
                               help="Negatif = turun, Positif = naik")

        if st.button("▶ Jalankan Simulasi", use_container_width=False):
            with st.spinner("Menghitung simulasi..."):
                wi = what_if_analysis(
                    model, scaler,
                    st.session_state.inputs,
                    feature_wi, pct_wi,
                )

            # ── SIDE-BY-SIDE ──────────────────────────────
            st.markdown("---")
            arrow = "⬇" if pct_wi < 0 else "⬆"
            st.markdown(
                f"### Hasil Simulasi: **{feature_wi}** {arrow} {abs(pct_wi)}%\n"
                f"Nilai berubah dari `{wi['original_val']}` → `{wi['modified_val']}`"
            )

            bc_col, mid_col, ac_col = st.columns([5, 1, 5])
            with bc_col:
                scat = {"Tinggi": "🟢", "Sedang": "🟡", "Rendah": "🔴"}
                st.markdown(f"""
                <div class="compare-before">
                  <h4 style="color:#f87171;">SEBELUM</h4>
                  <p>🌡️ Heating Load<br><b style="font-size:1.6rem;">{wi['before_hl']}</b> kWh/m²</p>
                  <p>❄️ Cooling Load<br><b style="font-size:1.6rem;">{wi['before_cl']}</b> kWh/m²</p>
                  <p>⚡ Efficiency Score<br><b style="font-size:1.8rem;">{wi['before_score']}</b>/100</p>
                  <p>{scat.get(wi['before_cat'], '🔵')} {wi['before_cat']}</p>
                </div>
                """, unsafe_allow_html=True)

            with mid_col:
                st.markdown("<br><br><br><center>→</center>", unsafe_allow_html=True)

            with ac_col:
                st.markdown(f"""
                <div class="compare-after">
                  <h4 style="color:#4ade80;">SESUDAH</h4>
                  <p>🌡️ Heating Load<br><b style="font-size:1.6rem;">{wi['after_hl']}</b> kWh/m²</p>
                  <p>❄️ Cooling Load<br><b style="font-size:1.6rem;">{wi['after_cl']}</b> kWh/m²</p>
                  <p>⚡ Efficiency Score<br><b style="font-size:1.8rem;">{wi['after_score']}</b>/100</p>
                  <p>{scat.get(wi['after_cat'], '🔵')} {wi['after_cat']}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── DELTA METRICS ─────────────────────────────
            st.markdown("---")
            st.markdown("#### 📉 Perubahan yang Terjadi")
            d1, d2, d3 = st.columns(3)
            with d1:
                delta_hl = wi['delta_hl']
                st.metric("Heating Load", f"{wi['after_hl']} kWh/m²",
                          delta=f"{delta_hl:+.2f} kWh/m²",
                          delta_color="inverse")
            with d2:
                delta_cl = wi['delta_cl']
                st.metric("Cooling Load", f"{wi['after_cl']} kWh/m²",
                          delta=f"{delta_cl:+.2f} kWh/m²",
                          delta_color="inverse")
            with d3:
                delta_sc = wi['delta_score']
                st.metric("Efficiency Score", f"{wi['after_score']}/100",
                          delta=f"{delta_sc:+d} poin")

            # ── INTERPRETASI ──────────────────────────────
            direction = "menurun" if (delta_hl + delta_cl) < 0 else "meningkat"
            total_delta_pct = abs((delta_hl + delta_cl) / max((wi['before_hl'] + wi['before_cl']), 0.01) * 100)
            st.info(
                f"💡 Dengan mengubah **{feature_wi}** sebesar **{pct_wi:+d}%**, "
                f"total beban energi diperkirakan **{direction}** sekitar **{total_delta_pct:.1f}%**. "
                f"Efficiency Score berubah dari **{wi['before_score']}** → **{wi['after_score']}** "
                f"(**{delta_sc:+d} poin**)."
            )


# ══════════════════════════════════════════════════════════
# TAB 4 — CRISP-DM
# ══════════════════════════════════════════════════════════
with tab_crisp:
    st.markdown('<div class="section-header">📋 Metodologi CRISP-DM</div>', unsafe_allow_html=True)

    phases = [
        (
            "1️⃣ Business Understanding",
            """
Bangunan merupakan konsumen energi terbesar di dunia (IEA, 2023). Sebagian besar konsumsi
energi bangunan berasal dari kebutuhan pendinginan (*Cooling Load*) dan pemanasan (*Heating Load*)
yang ditentukan oleh karakteristik fisik sejak tahap perancangan.

**Tujuan Bisnis:**
- Membantu arsitek dan perancang bangunan melakukan estimasi kebutuhan energi sejak tahap desain awal.
- Memberikan rekomendasi desain berbasis data untuk meningkatkan efisiensi energi sebelum konstruksi.
- Mengurangi potensi biaya renovasi akibat desain yang kurang efisien.
- Mendukung pengambilan keputusan dalam perancangan bangunan berkelanjutan (*sustainable building*).
- Meningkatkan transparansi hasil prediksi melalui **Explainable AI**.
            """
        ),
        (
            "2️⃣ Data Understanding",
            """
**Dataset:** UCI Energy Efficiency Dataset
- **Sumber:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
- **Sampel:** 768 bangunan
- **Fitur Input (8):** Relative Compactness, Surface Area, Wall Area, Roof Area,
  Overall Height, Orientation, Glazing Area, Glazing Area Distribution
- **Target Output (2):** Heating Load (Y1), Cooling Load (Y2)

**Analisis Korelasi:**
Dilakukan menggunakan korelasi Pearson dan visualisasi Heatmap untuk memahami hubungan antar variabel.
Fitur seperti *Relative Compactness*, *Roof Area*, dan *Glazing Area* memiliki korelasi tinggi
terhadap kedua target.
            """
        ),
        (
            "3️⃣ Data Preparation",
            """
**Langkah-langkah:**
1. **Validasi Data** — Pengecekan nilai kosong, duplikat, dan nilai tidak valid.
2. **Normalisasi** — Menggunakan `StandardScaler` agar semua fitur berada pada skala yang sama.
3. **Split Data** — Pembagian 70% data latih dan 30% data uji (`train_test_split`, `random_state=42`).

**Hasil:**
- Data latih: 537 sampel
- Data uji: 231 sampel
- Tidak ditemukan nilai kosong atau duplikat pada dataset UCI.
            """
        ),
        (
            "4️⃣ Modeling",
            """
**Algoritma:** `MultiOutputRegressor(RandomForestRegressor)`

**Alasan Pemilihan:**
- Akurasi tinggi pada data tabular non-linear
- Tahan terhadap overfitting dibanding Decision Tree tunggal
- Mendukung **Feature Importance** untuk Explainable AI
- `MultiOutputRegressor` memungkinkan prediksi dua target sekaligus

**Parameter Model:**
```
RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
```
            """
        ),
        (
            "5️⃣ Evaluation",
            """
**Metrik Evaluasi:**

| Metrik | Keterangan |
|--------|------------|
| **MAPE** | Mean Absolute Percentage Error — tingkat kesalahan dalam persentase |
| **RMSE** | Root Mean Squared Error — sensitivitas terhadap kesalahan besar |
| **R² Score** | Koefisien determinasi — seberapa baik model menjelaskan variasi data |

MAPE digunakan sebagai metrik utama karena lebih mudah dipahami pengguna non-teknis seperti arsitek.
Nilai MAPE < 10% menunjukkan model yang sangat baik untuk kasus ini.

Jalankan `python train_model.py` untuk melihat hasil evaluasi lengkap pada terminal.
            """
        ),
        (
            "6️⃣ Interpretation & Recommendation",
            """
**Tahap Tambahan CRISP-DM untuk Proyek Ini:**

Hasil prediksi Heating Load dan Cooling Load diterjemahkan menjadi **tingkat efisiensi energi bangunan**
menggunakan Energy Efficiency Score (0–100):

| Score | Kategori | Interpretasi |
|-------|----------|--------------|
| 80–100 | 🟢 Tinggi | Desain sangat efisien, kebutuhan energi rendah |
| 60–79  | 🟡 Sedang | Ada ruang perbaikan, perhatikan rekomendasi |
| < 60   | 🔴 Rendah | Bangunan boros energi, perlu revisi desain |

Sistem kemudian menghasilkan **rekomendasi perbaikan desain** berdasarkan fitur yang paling
berpengaruh terhadap hasil prediksi (*Feature Importance*). Tahap ini bertujuan membantu pengguna
mengambil keputusan desain yang lebih efisien dan berkelanjutan.

**Fitur Transparansi:**
- Performance Map (P-Map) — visualisasi posisi bangunan terhadap zona efisiensi
- Explainable AI — kontribusi setiap fitur dalam persentase
- What-If Analysis — simulasi dampak perubahan parameter desain
- Estimasi Biaya — proyeksi konsumsi dan biaya listrik bulanan
            """
        ),
    ]

    for title, content in phases:
        with st.expander(title, expanded=False):
            st.markdown(content)

    st.markdown("---")
    st.markdown("""
    <div class="info-box" style="text-align:center;">
      <strong>EnergyPredict AI</strong> — Sistem Prediksi Efisiensi Energi Bangunan<br>
      Menggunakan Random Forest Regressor · MultiOutputRegressor · Streamlit<br>
      Dataset: UCI Energy Efficiency (768 sampel)
    </div>
    """, unsafe_allow_html=True)