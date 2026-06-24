"""
app.py
======
Streamlit Application — Building Cooling and Heating Load Prediction & Calculation

Features:
  1. Building Design Input (Sidebar)
  2. Input Validation
  3. Prediction (Heating Load & Cooling Load)
  4. Energy Efficiency Score
  5. Performance Map (P-Map)
  6. Explainable AI (Feature Importance)
  7. Automated Recommendations
  8. What-If Analysis

Run:
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
    get_local_explanation,
    get_recommendation,
    make_pmap_figure,
    make_importance_figure,
    what_if_analysis,
    FEATURE_NAMES,
    TARGET_NAMES,  
)

st.set_page_config(
    page_title="ArchiFlux AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        color-scheme: dark !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght=0,200..800;1,200..800&family=Inter:wght@300..700&display=swap');
          
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #2A3324 0%,
            #31452B 50%,
            #3A5231 100%
        );
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #EFE5D1 0%,
            #E8DDC7 100%
        );
        border-right: 2px solid #E6A59E;
        color: #5A4A42 !important;
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
        gap: 0.5rem !important;
    }

    [data-testid="stSidebarUserContent"] hr {
        margin: 0.8rem 0 !important;
        border-color: rgba(234,156,175,0.25) !important;
    }

    [data-testid="stSidebar"] label {
        color: #5A4A42 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] p {
        color: #5A4A42 !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        color: #3D7D3D !important;
        background-color: transparent !important;
    }
    [data-testid="collapsedControl"] button {
        color: #3D7D3D !important;
    }

    .main-header {
        background: linear-gradient(
            135deg,
            rgba(230,165,158,0.18) 0%,
            rgba(180,212,122,0.18) 100%
        );
        border: 1px solid rgba(230,165,158,0.35);
        border-radius: 16px !important; /* Mengeliminasi kelancipan sudut */
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
                
    .main-header h1 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #EFE5D1;
        letter-spacing: 1px;
        margin: 0;
    }

    .main-header p.sub-title { 
        color: #EFE5D1; 
        margin: 0.2rem 0 0 0; 
        font-size: 1.05rem; 
        font-weight: 500;
    }
    
    .main-header p.meta-info { 
        color: #A3A3A3; 
        margin: 0.3rem 0 0 0; 
        font-size: 0.8rem; 
    }

    .stTabs {
        margin-top: 1.5rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: #EFE5D1;
        color: #5A4A42;
        border-radius: 12px 12px 0 0;
        margin-right: 6px;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: #3D7D3D !important;
        color: white !important;
    }

    .metric-card {
        background: rgba(44,58,38,0.85);
        border: 1px solid rgba(180,212,122,0.35);
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

    .score-badge {
        display: inline-block;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }

    .section-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #B4D47A;
        border-left: 4px solid #3D7D3D;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    .info-box {
        background: rgba(239,229,209,0.06);
        border: 1px solid rgba(180,212,122,0.25);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #F3EEF1;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .rec-card {
        background: linear-gradient(
            135deg,
            rgba(230,165,158,0.12) 0%,
            rgba(180,212,122,0.12) 100%
        );
        border: 1px solid rgba(230,165,158,0.3);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 0.6rem 0;
        color: #F3EEF1;
        line-height: 1.7;
    }

    .compare-before {
        background: rgba(230,165,158,0.10);
        border: 1px solid rgba(230,165,158,0.35);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .compare-after {
        background: rgba(180,212,122,0.10);
        border: 1px solid rgba(180,212,122,0.35);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }

    .stNumberInput input {
        background: rgba(243,238,241,0.08) !important;
        color: #F3EEF1 !important;
        border: 1px solid rgba(234,156,175,0.3) !important;
    }

    .stSelectbox *, [data-baseweb="select"] > div, [data-baseweb="select"] span, [data-baseweb="select"] input {
        color: #F3EEF1 !important;
    }

    div[role="listbox"] {
        background-color: #24311F !important;
    }

    div[role="option"] {
        color: #F3EEF1 !important;
    }

    .stButton > button {
        background: linear-gradient(
            135deg,
            #B4D47A 0%,
            #3D7D3D 100%
        );
        color: white;
        border: none;
        border-radius: 12px;
    }
    .stButton > button:hover { 
        opacity: 0.9; 
    }

    #MainMenu, footer { 
        visibility: hidden; 
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    return load_model()


model, scaler, feature_names = get_model()
model_loaded = model is not None


st.markdown("""
<div class="main-header">
  <h1>ArchiFlux AI</h1>
  <p>Predict & Calculate Building Heating Load · Cooling Load · Energy Efficiency</p>
    Based on UCI Energy Efficiency Dataset · Random Forest Regressor · MultiOutputRegressor
  </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "**Model not available.** Please run `python train_model.py` first "
        "to train and save the model, then refresh this page."
    )
    st.stop()


with st.sidebar:
    st.markdown("Building Design Input")
    st.markdown("---")

    st.markdown("#### Geometric Characteristics")
    rc  = st.slider("Relative Compactness",  0.01, 1.00, 0.76, 0.01,
                    help="Ratio of building volume to surface area (0–1)")
    sa  = st.number_input("Surface Area (m²)",    min_value=1.0, max_value=2000.0, value=661.5, step=1.0)
    wa  = st.number_input("Wall Area (m²)",        min_value=1.0, max_value=2000.0, value=318.5, step=1.0)
    ra  = st.number_input("Roof Area (m²)",        min_value=1.0, max_value=2000.0, value=122.5, step=0.5)
    oh  = st.selectbox("Overall Height (m)",    [3.5, 7.0], index=1,
                       help="Select building height: 3.5m (1 floor) or 7.0m (2 floors)")

    st.markdown("---")
    st.markdown("#### Orientation & Glazing")
    orient_labels = {2: "2 — North", 3: "3 — East", 4: "4 — South", 5: "5 — West"}
    orient_sel = st.selectbox("Orientation", list(orient_labels.keys()),
                              format_func=lambda x: orient_labels[x],
                              help="Facing direction of the main building facade")
    ga  = st.slider("Glazing Area (ratio)", 0.00, 1.00, 0.25, 0.05,
                    help="Proportion of window area to total facade area (0–1)")
    gad_labels = {0: "0 — No glazing", 1: "1 — Uniform", 2: "2 — N>E=W>S",
                  3: "3 — E>W=N>S", 4: "4 — S>E=W>N", 5: "5 — S>N=E>W"}
    gad = st.selectbox("Glazing Area Distribution", list(gad_labels.keys()),
                       format_func=lambda x: gad_labels[x],
                       help="Distribution of glazing across building sides")

    st.markdown("---")
    predict_btn = st.button("Predict Now", use_container_width=True)


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



if "result" not in st.session_state:
    st.session_state.result = None
if "fi_df" not in st.session_state:
    st.session_state.fi_df = None


if predict_btn:
    is_valid, errors = validate_inputs(user_inputs)
    if not is_valid:
        for err in errors:
            st.error(f"{err}")
        st.session_state.result = None
    else:
        with st.spinner("Running AI prediction..."):
            result = predict(model, scaler, user_inputs)
            fi_df = get_local_explanation(
                model,
                scaler,
                user_inputs
            )
            st.session_state.result    = result
            st.session_state.fi_df    = fi_df
            st.session_state.inputs   = user_inputs.copy()
        st.success("Prediction successful!")


tab_pred, tab_xai, tab_whatif = st.tabs([
    "Prediction Results",
    "Explainable AI",
    "What-If Analysis",
])


with tab_pred:
    if st.session_state.result is None:
        st.markdown("""
        <div class="info-box">
          Configure the building design parameters in the left sidebar, then press
          <strong>Predict Now</strong> to view the energy analysis results.
        </div>
        """, unsafe_allow_html=True)

        # Blank P-Map demo preview
        st.markdown('<div class="section-header">Performance Map (Preview)</div>', unsafe_allow_html=True)
        fig_demo = make_pmap_figure()
        st.plotly_chart(fig_demo, use_container_width=True)

    else:
        res    = st.session_state.result
        fi_df  = st.session_state.fi_df
        hl     = res["Heating Load"]
        cl     = res["Cooling Load"]
        score, cat, color_hex = energy_efficiency_score(hl, cl)
        
        st.markdown('<div class="section-header">Energy Prediction Results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Heating Load</div>
              <div class="metric-value">{hl}</div>
              <div class="metric-unit">kWh/m²</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Cooling Load</div>
              <div class="metric-value">{cl}</div>
              <div class="metric-unit">kWh/m²</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            score_bg = {"High": "rgba(34,197,94,0.15)", "Medium": "rgba(245,158,11,0.15)", "Low": "rgba(239,68,68,0.15)"}.get(cat, "rgba(34,197,94,0.15)")
            st.markdown(f"""
            <div class="metric-card" style="background:{score_bg};">
              <div class="metric-label">Efficiency Score</div>
              <div class="metric-value" style="color:{color_hex};">{score}</div>
              <div class="metric-unit">/ 100 — {cat}</div>
            </div>""", unsafe_allow_html=True)


        st.markdown("---")
        score_desc = {
            "High": f"**Highly efficient building!** With a score of **{score}/100**, this design falls into the **High** efficiency category. The energy demand for this building is low and well-optimized.",
            "Medium": f"**Moderate building efficiency.** A score of **{score}/100** indicates room for improvement. Review the recommendations under the XAI tab to enhance energy performance.",
            "Low": f"**Energy-inefficient building!** A score of **{score}/100** falls into the **Low** efficiency category. Consider revising design parameters prior to construction to minimize long-term operational costs.",
        }
        st.info(score_desc.get(cat, f"Building falls into the {cat} efficiency category."))

        st.markdown('<div class="section-header">Performance Map</div>', unsafe_allow_html=True)
        fig_pmap = make_pmap_figure(hl, cl)
        st.plotly_chart(fig_pmap, use_container_width=True)


        with st.expander("Summary of Input Parameters"):
            inp_df = pd.DataFrame({
                "Feature": list(st.session_state.inputs.keys()),
                "Value": [round(v, 4) for v in st.session_state.inputs.values()],
            })
            st.dataframe(inp_df, use_container_width=True, hide_index=True)


with tab_xai:
    st.markdown('<div class="section-header">Explainable AI — Feature Importance</div>', unsafe_allow_html=True)

    if st.session_state.fi_df is None:
        st.info("Run the prediction first to display the Explainable AI analysis.")
    else:
        fi_df = st.session_state.fi_df
        res   = st.session_state.result
        hl    = res["Heating Load"]
        cl    = res["Cooling Load"]
        score, cat, color_hex = energy_efficiency_score(hl, cl)

        top1  = fi_df.iloc[0]
        top1_pct = top1["Pct"]
        st.markdown(f"""
        <div class="info-box">
          The factor that most influences your building's energy prediction is
          <strong>{top1['Feature']}</strong> with a contribution of <strong>{top1_pct:.1f}%</strong>.
          Modifying this specific aspect will yield the highest impact on energy efficiency.
        </div>
        """, unsafe_allow_html=True)

        fig_fi = make_importance_figure(fi_df)
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("Detailed Feature Importance Table"):
            display_fi = fi_df[["Feature", "Pct"]].copy()
            display_fi.columns = ["Building Feature", "Contribution (%)"]
            st.dataframe(display_fi, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Automated Design Recommendations</div>', unsafe_allow_html=True)
        recs = get_recommendation(fi_df)
        for rec in recs:
            rank_label = f"Rank {rec['rank']}"
            with st.expander(f"{rank_label}: {rec['feature']} — Contribution {rec['pct']:.1f}%", expanded=(rec["rank"] == 1)):
                st.markdown(f'<div class="rec-card">{rec["text"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Analytical Breakdown</div>', unsafe_allow_html=True)
        top_features_str = ", ".join([f["feature"] for f in recs[:2]])
        avg_load = (hl + cl) / 2
        avg_ref  = 22.0  # UCI dataset average reference
        diff_pct = ((avg_load - avg_ref) / avg_ref) * 100

        if diff_pct > 0:
            math_text = (
                f"Your building belongs to the {cat} efficiency zone because the combined effect of "
                f"high {top_features_str} leads to an average energy load of "
                f"{avg_load:.1f} kWh/m², which is {abs(diff_pct):.1f}% higher "
                f"than the average efficient building in the UCI dataset ({avg_ref} kWh/m²)."
            )
        else:
            math_text = (
                f"Your building belongs to the {cat} efficiency zone. The average energy load of "
                f"{avg_load:.1f} kWh/m² is {abs(diff_pct):.1f}% lower "
                f"than the UCI dataset benchmark baseline ({avg_ref} kWh/m²). This design is highly optimal!"
            )

        st.markdown(f'<div class="info-box">Note: {math_text}</div>', unsafe_allow_html=True)

with tab_whatif:
    st.markdown('<div class="section-header">What-If Analysis</div>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.info("Run the prediction first to utilize the What-If Analysis workspace.")
    else:
        st.markdown("""
        <div class="info-box">
          Simulate changes to an individual design parameter to instantly assess its impact on
          Heating Load, Cooling Load, and the cumulative Energy Efficiency Score.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        wi_col1, wi_col2 = st.columns([2, 1])
        with wi_col1:
            feature_wi = st.selectbox(
                "Select Feature to Modify",
                FEATURE_NAMES,
                index=FEATURE_NAMES.index("Glazing Area"),
            )
        with wi_col2:
            pct_wi = st.slider("Percentage Change (%)", -50, 50, -10, 5,
                               help="Negative values decrease; positive values increase the parameter value")

        if st.button("Run Simulation", use_container_width=False):
            with st.spinner("Calculating simulation updates..."):
                wi = what_if_analysis(
                    model, scaler,
                    st.session_state.inputs,
                    feature_wi, pct_wi,
                )

            # SIDE-BY-SIDE COMPARISON 
            st.markdown("---")
            direction_arrow = "Decreased by" if pct_wi < 0 else "Increased by"
            st.markdown(
                f"### Simulation Results: **{feature_wi}** {direction_arrow} {abs(pct_wi)}%\n"
                f"Value shifted from `{wi['original_val']}` → `{wi['modified_val']}`"
            )

            bc_col, mid_col, ac_col = st.columns([5, 1, 5])
            with bc_col:
                scat = {"High": "Green", "Medium": "Yellow", "Low": "Red"}
                st.markdown(f"""
                <div class="compare-before">
                  <h4 style="color:#f87171;">BEFORE</h4>
                  <p>Heating Load<br><b style="font-size:1.6rem;">{wi['before_hl']}</b> kWh/m²</p>
                  <p>Cooling Load<br><b style="font-size:1.6rem;">{wi['before_cl']}</b> kWh/m²</p>
                  <p>Efficiency Score<br><b style="font-size:1.8rem;">{wi['before_score']}</b>/100</p>
                  <p>{scat.get(wi['before_cat'], 'Status')}: {wi['before_cat']}</p>
                </div>
                """, unsafe_allow_html=True)

            with mid_col:
                st.markdown("<br><br><br><center>→</center>", unsafe_allow_html=True)

            with ac_col:
                st.markdown(f"""
                <div class="compare-after">
                  <h4 style="color:#4ade80;">AFTER</h4>
                  <p>Heating Load<br><b style="font-size:1.6rem;">{wi['after_hl']}</b> kWh/m²</p>
                  <p>Cooling Load<br><b style="font-size:1.6rem;">{wi['after_cl']}</b> kWh/m²</p>
                  <p>Efficiency Score<br><b style="font-size:1.8rem;">{wi['after_score']}</b>/100</p>
                  <p>{scat.get(wi['after_cat'], 'Status')}: {wi['after_cat']}</p>
                </div>
                """, unsafe_allow_html=True)

            # DELTA METRICS 
            st.markdown("---")
            st.markdown("#### Observed Variances")
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
                          delta=f"{delta_sc:+d} points")

            # INTERPRETATION 
            direction = "decreased" if (delta_hl + delta_cl) < 0 else "increased"
            total_delta_pct = abs((delta_hl + delta_cl) / max((wi['before_hl'] + wi['before_cl']), 0.01) * 100)
            st.info(
                f"Analysis: By modifying **{feature_wi}** by **{pct_wi:+d}%**, "
                f"total energy demand is projected to be **{direction}** by roughly **{total_delta_pct:.1f}%**. "
                f"The Efficiency Score shifted from **{wi['before_score']}** → **{wi['after_score']}** "
                f"({delta_sc:+d} points)."
            )