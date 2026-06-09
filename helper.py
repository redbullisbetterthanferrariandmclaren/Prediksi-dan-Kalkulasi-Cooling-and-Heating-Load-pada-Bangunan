"""
helper.py
=========
Helper functions for the Streamlit building Heating Load & Cooling Load 
prediction application.

Contains:
  - load_model()             : loads the model, scaler, and feature names
  - validate_inputs()        : validates user form inputs
  - predict()                : runs multi-output predictions
  - energy_efficiency_score(): calculates 0-100 score and efficiency category
  - get_feature_importance() : calculates feature contribution percentages
  - get_recommendation()     : generates recommendations based on dominant features
  - make_pmap_figure()       : creates Plotly Performance Map (P-Map)
  - make_importance_figure(): creates Plotly Feature Importance bar chart
  - what_if_analysis()       : simulates changes to a single feature
  - estimate_cost()          : estimates monthly operational costs
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import shap

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

DEFAULT_PLN_RATE = 1_444   
BUILDING_AREA_M2 = 100     


def load_model():
    """Loads the model, scaler, and feature names from disk."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURE_PATH]):
        return None, None, None

    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, scaler, features


def validate_inputs(inputs: dict) -> tuple[bool, list[str]]:
    """
    Validates the building inputs dictionary.
    Returns (is_valid: bool, errors: list[str]).
    """
    errors = []

    rc = inputs.get("Relative Compactness", -1)
    if not (0.0 < rc <= 1.0):
        errors.append("Relative Compactness must be between 0 (exclusive) and 1.")

    area_fields = ["Surface Area", "Wall Area", "Roof Area"]
    for field in area_fields:
        val = inputs.get(field, 0)
        if val <= 0:
            errors.append(f"{field} must be greater than 0.")

    for key, val in inputs.items():
        if val < 0:
            errors.append(f"{key} cannot be a negative value.")

    ga = inputs.get("Glazing Area", -1)
    if not (0.0 <= ga <= 1.0):
        errors.append("Glazing Area must be between 0 and 1.")

    return (len(errors) == 0), errors


def predict(model, scaler, inputs: dict) -> dict:
    """
    Runs predictions for both Heating Load and Cooling Load.
    inputs : dict with keys matching FEATURE_NAMES
    returns: dict {"Heating Load": float, "Cooling Load": float}
    """
    x = np.array([[inputs[f] for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return {
        "Heating Load": round(float(pred[0]), 6),
        "Cooling Load": round(float(pred[1]), 6),
    }


def energy_efficiency_score(hl: float, cl: float) -> tuple[int, str, str]:
    """
    Calculates the Energy Efficiency Score (0-100) and its category.

    Approach:
      - Average HL and CL from the UCI dataset ≈ 22 kWh/m²
      - Maximum reference efficiency value = 10 kWh/m² (Highly efficient)
      - Minimum reference efficiency value = 45 kWh/m² (Highly inefficient)

    Returns: (score: int, category: str, color: str)
    """
    avg = (hl + cl) / 2
    # Normalization: score 100 if avg <= 10, score 0 if avg >= 45
    score = int(np.clip((45 - avg) / (45 - 10) * 100, 0, 100))

    if score >= 80:
        return score, "High", "#22c55e"
    elif score >= 60:
        return score, "Medium", "#f59e0b"
    else:
        return score, "Low", "#ef4444"


def get_feature_importance(model) -> pd.DataFrame:
    """
    Extracts the mean feature importance across all estimators.
    Returns a DataFrame with columns: Feature, Importance, Pct
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
    Computes instance-level feature importance customized to the user's inputs.
    """
    x = np.array([[inputs[f] for f in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)

    # Use the first estimator (Heating Load) for local explanation
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


RECOMMENDATION_MAP = {
    "Relative Compactness": (
        "Compact the Building Design\n\n"
        "Relative Compactness is a dominant factor in this prediction. "
        "A more compact building minimizes the surface area exposed to the "
        "external environment, reducing unwanted energy loss.\n\n"
        "→ Consider a tighter building footprint and avoid configurations that "
        "are excessively elongated or complex."
    ),
    "Surface Area": (
        "Optimize Total Surface Area\n\n"
        "A larger overall surface area expands the envelope available for heat exchange "
        "between the building and the outdoors. This increases both cooling and heating loads.\n\n"
        "→ Reduce the total surface envelope area or enhance insulation specifications "
        "across all exterior building assemblies."
    ),
    "Wall Area": (
        "Enhance Wall Insulation\n\n"
        "Wall Area is the most significant structural driver in this scenario. "
        "Expansive wall surface areas or subpar thermal insulation drastically accelerate "
        "heat transfer rates.\n\n"
        "→ Deploy building materials with superior thermal resistance (R-values), "
        "such as autoclaved aerated concrete (AAC), sandwich panels, or continuous mineral wool cavity insulation."
    ),
    "Roof Area": (
        "Optimize Roof Design\n\n"
        "The roof receives the highest density of direct solar radiation. "
        "A vast roof area combined with poor insulation metrics will spike the Cooling Load "
        "substantially.\n\n"
        "→ Explore roof area reductions, apply continuous reflective cool roof coatings, "
        "or integrate solar photovoltaic arrays to shade the roof deck."
    ),
    "Overall Height": (
        "Evaluate Building Height\n\n"
        "Building height determines the aggregate volume of conditioned air spaces "
        "and dictates exposure profiles to wind loads.\n\n"
        "→ Establish optimal floor-to-ceiling heights and verify that the HVAC air-side "
        "distribution systems are sized strictly to the calculated volumetric space."
    ),
    "Orientation": (
        "Optimize Building Orientation\n\n"
        "Building orientation sets the localized solar radiation loads impacting the facade. "
        "An unfavorable orientation can induce excessive peak cooling demands.\n\n"
        "→ Align primary facades containing major glazing installations toward the North or South "
        "to limit direct early morning and late afternoon solar heat gains."
    ),
    "Glazing Area": (
        "Reduce Glazing Area or Upgrade Glass Specifications\n\n"
        "Glazing Area is the LARGEST single contributor to this analytical model. "
        "Standard monolithic glass features poor thermal resistance, creating massive "
        "thermal bridges via high U-values.\n\n"
        "→ Restrict the total window-to-wall ratio, or transition to high-performance low-e, "
        "double-glazed, or triple-glazed setups to cut heat transfer indices up to 50%."
    ),
    "Glazing Area Distribution": (
        "Distribute Glazing Strategically\n\n"
        "The spatial distribution of windows across various building faces heavily alters thermal efficiency. "
        "Concentrating windows on East or West exposures maximizes intense solar heat stress.\n\n"
        "→ Distribute glazing apertures uniformly or prioritize concentrations on North and South facades "
        "to manage passive solar heat gain values systematically."
    ),
}


def get_recommendation(feature_importance_df: pd.DataFrame) -> list[dict]:
    """
    Generates a structured list of recommendations mapped to the top 4 Feature Importance metrics.
    Returns: list of dict {"rank", "feature", "pct", "text"}
    """
    recs = []
    for rank, row in feature_importance_df.head(4).iterrows():
        feat = row["Feature"]
        text = RECOMMENDATION_MAP.get(feat, "Monitor design constraints for this feature.")
        recs.append({
            "rank": rank + 1,
            "feature": feat,
            "pct": row["Pct"],
            "text": text,
        })
    return recs


def make_pmap_figure(hl_user: float = None, cl_user: float = None) -> go.Figure:
    """
    Generates a Performance Map (P-Map) illustrating localized efficiency zones 
    and mapping the user's specific building metrics.
    """
    # Efficiency zone grid boundaries
    hl_range = np.linspace(5, 50, 200)
    cl_range = np.linspace(5, 50, 200)
    HL_grid, CL_grid = np.meshgrid(hl_range, cl_range)
    avg_grid = (HL_grid + CL_grid) / 2

    # Map grid scores
    score_grid = np.clip((45 - avg_grid) / (45 - 10) * 100, 0, 100)

    fig = go.Figure()

    # Efficiency zones contour layer
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
            ticktext=["0", "20", "40 Low", "60", "80 Medium", "100 High"],
        ),
        opacity=0.75,
        name="Efficiency Zones",
        hovertemplate="HL=%{x:.1f}  CL=%{y:.1f}  Score=%{z:.0f}<extra></extra>",
    ))

    # Reference dataset markers (random sample scatter plot)
    np.random.seed(0)
    n_ref = 80
    hl_ref = np.random.uniform(6, 45, n_ref)
    cl_ref = np.random.uniform(6, 45, n_ref)
    fig.add_trace(go.Scatter(
        x=hl_ref, y=cl_ref,
        mode="markers",
        marker=dict(size=5, color="rgba(255,255,255,0.35)", line=dict(width=1, color="white")),
        name="Dataset Reference",
        hovertemplate="HL=%{x:.1f}  CL=%{y:.1f}<extra>Reference</extra>",
    ))

    # User building placement marker
    if hl_user is not None and cl_user is not None:
        score_user, cat, color = energy_efficiency_score(hl_user, cl_user)
        fig.add_trace(go.Scatter(
            x=[hl_user], y=[cl_user],
            mode="markers+text",
            marker=dict(size=18, color=color, symbol="star", line=dict(width=2, color="white")),
            text=[f"  You\n({score_user}/100)"],
            textposition="middle right",
            textfont=dict(size=12, color="white"),
            name=f"Your Building ({cat})",
            hovertemplate=(
                f"<b>Your Building</b><br>"
                f"Heating Load : {hl_user} kWh/m²<br>"
                f"Cooling Load : {cl_user} kWh/m²<br>"
                f"Score        : {score_user}/100<br>"
                f"Category     : {cat}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="🗺️ Performance Map — Building Energy Efficiency Placement", font=dict(size=16)),
        xaxis=dict(title="Heating Load (kWh/m²)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Cooling Load (kWh/m²)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,40,0.85)",
        font=dict(color="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.75,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        height=520,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def make_importance_figure(fi_df: pd.DataFrame) -> go.Figure:
    """Creates a Feature Importance bar chart (Plotly)."""
    colors = px.colors.sequential.Viridis_r[:len(fi_df)]
    fig = go.Figure(go.Bar(
        x=fi_df["Feature"],
        y=fi_df["Pct"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in fi_df["Pct"]],
        textposition="outside",
        hovertemplate="%{x}<br>Contribution: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="📊 Feature Contribution toward Prediction (Feature Importance)",
        xaxis=dict(title="Building Features"),
        yaxis=dict(title="Contribution (%)", range=[0, fi_df["Pct"].max() * 1.25]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,40,0.85)",
        font=dict(color="white"),
        height=400,
        margin=dict(l=40, r=40, t=60, b=100),
    )
    fig.update_xaxes(tickangle=-30)
    return fig


def what_if_analysis(
    model,
    scaler,
    base_inputs: dict,
    feature_to_change: str,
    pct_change: float,
) -> dict:
    """
    Simulates dimensional changes to a single parameter.

    Parameters
    ----------
    base_inputs      : User's baseline configuration inputs
    feature_to_change: Target feature string name intended for modification
    pct_change       : Delta value expressed as percentage (e.g., -10 for a 10% decrease)

    Returns a dictionary containing delta evaluation keys:
        before_hl, before_cl, before_score, before_cat,
        after_hl, after_cl, after_score, after_cat,
        delta_hl, delta_cl, delta_score
    """
    # Evaluate baseline conditions
    before_pred = predict(model, scaler, base_inputs)
    bhl = before_pred["Heating Load"]
    bcl = before_pred["Cooling Load"]
    bscore, bcat, _ = energy_efficiency_score(bhl, bcl)

    # Perform structural modification
    modified = base_inputs.copy()
    original_val = modified[feature_to_change]
    new_val = original_val * (1 + pct_change / 100)

    # Enforce strict domain bounding conditions
    if feature_to_change == "Relative Compactness":
        new_val = np.clip(new_val, 0.01, 1.0)
    elif feature_to_change == "Glazing Area":
        new_val = np.clip(new_val, 0.0, 1.0)
    else:
        new_val = max(new_val, 0.01)

    modified[feature_to_change] = new_val

    # Evaluate scenario adjustments
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


def estimate_cost(
    hl: float,
    cl: float,
    area_m2: float = BUILDING_AREA_M2,
    pln_rate: float = DEFAULT_PLN_RATE,
) -> dict:
    """
    Estimates basic monthly utility outlays.

    Assumptions:
    - Consumption = (HL + CL) * area_m2 kWh/month
    - PLN Rate (Rp/kWh) can be configured dynamically by the user

    Returns: dict containing consumption_kwh and cost_rp metrics per month
    """
    total_load_kwh = (hl + cl) * area_m2          # kWh/month
    biaya_rp       = total_load_kwh * pln_rate     # Rp/month

    return {
        "consumption_kwh": round(total_load_kwh, 1),
        "cost_rp"        : round(biaya_rp, 0),
        "pln_rate"       : pln_rate,
        "area_m2"        : area_m2,
    }


def fmt_rp(value: float) -> str:
    """Formats numeric values to Indonesian Rupiah representation. Ex: 317680 -> 'Rp 317.680'"""
    return f"Rp {int(value):,}".replace(",", ".")