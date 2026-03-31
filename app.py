# streamlit_app.py
"""
Credit Invisibility Solver — Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json
import joblib
import os
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Invisibility Solver",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #1565C0; text-align: center; padding: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #1565C0, #42A5F5); border-radius: 12px;
                   padding: 1.2rem; color: white; text-align: center; }
    .risk-high   { background: #FFEBEE; border-left: 5px solid #F44336; padding: 1rem; border-radius: 8px; }
    .risk-medium { background: #FFF8E1; border-left: 5px solid #FF9800; padding: 1rem; border-radius: 8px; }
    .risk-low    { background: #E8F5E9; border-left: 5px solid #4CAF50; padding: 1rem; border-radius: 8px; }
    .sidebar-section { font-size: 0.9rem; color: #666; margin-bottom: 0.4rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Load artifacts ───────────────────────────────────────────────────────────
MODEL_DIR = "./models"

@st.cache_resource
def load_models():
    models = []
    for i in range(1, 6):
        path = f"{MODEL_DIR}/lgbm_fold_{i}.txt"
        if os.path.exists(path):
            m = lgb.Booster(model_file=path)
            models.append(m)
    return models

@st.cache_resource
def load_artifacts():
    pca    = joblib.load(f"{MODEL_DIR}/pca.pkl")     if os.path.exists(f"{MODEL_DIR}/pca.pkl")    else None
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")  if os.path.exists(f"{MODEL_DIR}/scaler.pkl") else None
    fc_path = f"{MODEL_DIR}/feature_cols.json"
    if os.path.exists(fc_path):
        with open(fc_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = []
    return pca, scaler, feature_cols

@st.cache_resource
def load_sbert():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

# ─── Helper functions ─────────────────────────────────────────────────────────
def build_single_applicant_features(inputs: dict, pca, sbert) -> pd.DataFrame:
    """Transform raw user inputs into model-ready features."""
    income  = inputs["income"]
    credit  = inputs["credit_amount"]
    age     = inputs["age"]
    emp_yrs = inputs["employment_years"]
    ext1    = inputs["ext_score_1"]
    ext2    = inputs["ext_score_2"]
    ext3    = inputs["ext_score_3"]
    
    # Build a synthetic text for NLP embedding
    literacy = "strong financial planning habits" if np.mean([ext1,ext2,ext3]) > 0.6 else (
               "moderate financial awareness" if np.mean([ext1,ext2,ext3]) > 0.4 else
               "limited financial experience")
    text = (
        f"Applicant aged {age:.0f} years with annual income of {income:.0f}. "
        f"Requesting credit of {credit:.0f}. Employed for {emp_yrs:.1f} years. "
        f"Client demonstrates {literacy}. External score: {np.mean([ext1,ext2,ext3]):.2f}. "
        f"{'Owns property.' if inputs['owns_realty'] else 'No property.'} "
        f"{'Has dependents.' if inputs['has_children'] else 'No children.'}"
    )
    
    # NLP embed + PCA
    if sbert is not None:
        emb = sbert.encode([text], normalize_embeddings=True)
        if pca is not None:
            emb = pca.transform(emb)
        nlp_dict = {f"NLP_EMB_{i}": emb[0][i] for i in range(emb.shape[1])}
    else:
        # Demo mode — deterministic pseudo-embeddings
        n_dims = pca.n_components_ if pca is not None else 32
        nlp_dict = {f"NLP_EMB_{i}": 0.0 for i in range(n_dims)}
    
    # Tabular features
    tab_dict = {
        "AMT_INCOME_TOTAL":       income,
        "AMT_CREDIT":             credit,
        "AMT_ANNUITY":            inputs["annuity"],
        "AMT_GOODS_PRICE":        credit * 0.9,
        "DAYS_BIRTH":             -age * 365,
        "DAYS_EMPLOYED":          -emp_yrs * 365,
        "EXT_SOURCE_1":           ext1,
        "EXT_SOURCE_2":           ext2,
        "EXT_SOURCE_3":           ext3,
        "EXT_SOURCE_MEAN":        np.mean([ext1, ext2, ext3]),
        "EXT_SOURCE_MIN":         np.min([ext1, ext2, ext3]),
        "EXT_SOURCE_PROD":        ext1 * ext2 * ext3,
        "EXT_SOURCE_STD":         np.std([ext1, ext2, ext3]),
        "EXT1_EXT2_INTERACTION":  ext1 * ext2,
        "EXT2_EXT3_INTERACTION":  ext2 * ext3,
        "CREDIT_INCOME_RATIO":    credit / (income + 1),
        "ANNUITY_INCOME_RATIO":   inputs["annuity"] / (income + 1),
        "CREDIT_TERM":            inputs["annuity"] / (credit + 1),
        "AGE_YEARS":              age,
        "EMPLOYMENT_YEARS":       emp_yrs,
        "EMPLOYED_RATIO":         emp_yrs / (age + 1),
        "INCOME_PER_PERSON":      income / (inputs["family_size"] + 1),
        "CNT_FAM_MEMBERS":        inputs["family_size"],
        "CNT_CHILDREN":           inputs["n_children"],
        "CHILDREN_RATIO":         inputs["n_children"] / (inputs["family_size"] + 1),
        "FLAG_OWN_REALTY":        int(inputs["owns_realty"]),
        "FLAG_OWN_CAR":           int(inputs["owns_car"]),
        "HAS_CAR_REALTY":         int(inputs["owns_realty"] and inputs["owns_car"]),
        "DOCUMENT_COUNT":         inputs["doc_count"],
        "TOTAL_ENQUIRIES":        inputs["total_enquiries"],
        "BUREAU_COUNT":           inputs["bureau_count"],
        "BUREAU_ACTIVE_COUNT":    inputs["bureau_active"],
    }
    
    feat = {**tab_dict, **nlp_dict}
    return pd.DataFrame([feat])

def predict_risk(df_feat: pd.DataFrame, models: list, feature_cols: list) -> float:
    """Ensemble predict across all loaded fold models."""
    # Align columns — fill missing with 0
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
    df_feat = df_feat[feature_cols]
    preds = [m.predict(df_feat, num_iteration=m.best_iteration) for m in models]
    return float(np.mean(preds))

def risk_band(score: float) -> tuple:
    if score < 0.15:
        return "LOW RISK", "risk-low", "#4CAF50", "✅"
    elif score < 0.40:
        return "MEDIUM RISK", "risk-medium", "#FF9800", "⚠️"
    else:
        return "HIGH RISK", "risk-high", "#F44336", "🚨"

def get_shap_values(model, df_feat, feature_cols):
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
    df_feat = df_feat[feature_cols]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(df_feat)
    if isinstance(sv, list):
        sv = sv[1]
    return sv, explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1], df_feat

# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">💳 Credit Invisibility Solver</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:1.1rem;'>"
        "Alternative data ML pipeline to score the 1.7B credit-invisible population"
        "</p>", unsafe_allow_html=True
    )
    st.divider()

    # Load models
    try:
        models       = load_models()
        pca, scaler, feature_cols = load_artifacts()
        sbert        = load_sbert()
        model_loaded = len(models) > 0
    except Exception as e:
        st.error(f"⚠️ Could not load models: {e}. Running in demo mode.")
        model_loaded = False
        models, pca, scaler, feature_cols = [], None, None, []

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.shields.io/badge/Model-LightGBM%20%2B%20XGBoost-brightgreen", use_container_width=True)
        st.markdown("### 🎛️ Applicant Profile")
        
        st.markdown('<div class="sidebar-section">Financial Info</div>', unsafe_allow_html=True)
        income         = st.number_input("Annual Income (₹)", 10000, 10000000, 250000, step=10000)
        credit_amount  = st.number_input("Requested Credit (₹)", 10000, 5000000, 500000, step=10000)
        annuity        = st.number_input("Monthly Annuity (₹)", 1000, 200000, 15000, step=1000)
        
        st.markdown('<div class="sidebar-section">Personal Info</div>', unsafe_allow_html=True)
        age            = st.slider("Age (years)", 20, 70, 35)
        employment_yrs = st.slider("Employment Years", 0, 40, 5)
        family_size    = st.slider("Family Size", 1, 10, 3)
        n_children     = st.slider("Number of Children", 0, 5, 0)
        
        st.markdown('<div class="sidebar-section">Assets</div>', unsafe_allow_html=True)
        owns_realty    = st.checkbox("Owns Property", True)
        owns_car       = st.checkbox("Owns Car", False)
        
        st.markdown('<div class="sidebar-section">Credit Bureau Signals</div>', unsafe_allow_html=True)
        ext_score_1    = st.slider("External Score 1 (Bureau)", 0.0, 1.0, 0.6, 0.01)
        ext_score_2    = st.slider("External Score 2 (Behaviour)", 0.0, 1.0, 0.55, 0.01)
        ext_score_3    = st.slider("External Score 3 (Alt Data)", 0.0, 1.0, 0.50, 0.01)
        bureau_count   = st.number_input("# Previous Bureau Enquiries", 0, 50, 2)
        bureau_active  = st.number_input("# Active Bureau Credits", 0, 20, 1)
        total_enquiries= st.number_input("# Total Loan Enquiries", 0, 100, 3)
        doc_count      = st.number_input("# Documents Submitted", 0, 20, 5)
        
        predict_btn    = st.button("🔮 Score Applicant", use_container_width=True, type="primary")

    # ── Main Panels ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    inputs = dict(
        income=income, credit_amount=credit_amount, annuity=annuity,
        age=age, employment_years=employment_yrs, family_size=family_size,
        n_children=n_children, owns_realty=owns_realty, owns_car=owns_car,
        ext_score_1=ext_score_1, ext_score_2=ext_score_2, ext_score_3=ext_score_3,
        bureau_count=bureau_count, bureau_active=bureau_active,
        total_enquiries=total_enquiries, doc_count=doc_count, has_children=n_children>0,
    )

    if predict_btn or True:  # Show demo on load
        with st.spinner("Running ML pipeline..."):
            df_feat = build_single_applicant_features(inputs, pca, sbert)
            
            if model_loaded:
                risk_score = predict_risk(df_feat, models, feature_cols)
            else:
                # Demo mode — compute heuristic score
                risk_score = float(np.clip(
                    0.9 - 0.4*np.mean([ext_score_1,ext_score_2,ext_score_3])
                    - 0.1*(employment_yrs/40)
                    + 0.15*(credit_amount/income if income>0 else 0.5)
                    + np.random.normal(0, 0.02),
                    0.01, 0.99
                ))
        
        label, css_class, color, icon = risk_band(risk_score)
        credit_score = int(300 + (1 - risk_score) * 550)  # map to 300-850 range
        
        # ── KPI Row ───────────────────────────────────────────────────────────
        col1.metric("Default Probability", f"{risk_score*100:.1f}%", delta=f"{(risk_score-0.5)*100:+.1f}% vs avg")
        col2.metric("Alt Credit Score",    f"{credit_score}", delta=None)
        col3.metric("Risk Band",           f"{icon} {label}",  delta=None)
        
        st.divider()
        
        # ── Risk Card ─────────────────────────────────────────────────────────
        st.markdown(f'<div class="{css_class}"><b>{icon} Risk Assessment: {label}</b><br>'
                    f'Default probability: <b>{risk_score*100:.1f}%</b> | '
                    f'Alternative credit score: <b>{credit_score}/850</b></div>',
                    unsafe_allow_html=True)
        
        st.divider()
        
        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Breakdown", "🔍 SHAP Explainability", "📉 Drift Simulation", "📋 Feature Profile"])
        
        with tab1:
            c1, c2 = st.columns(2)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=credit_score,
                delta={"reference": 650, "valueformat": ".0f"},
                title={"text": "Alternative Credit Score", "font": {"size": 18}},
                gauge={
                    "axis":     {"range": [300, 850]},
                    "bar":      {"color": color},
                    "steps":    [
                        {"range": [300, 550], "color": "#FFEBEE"},
                        {"range": [550, 650], "color": "#FFF8E1"},
                        {"range": [650, 750], "color": "#E8F5E9"},
                        {"range": [750, 850], "color": "#C8E6C9"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 650},
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
            c1.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk factor radar
            categories = ["External Scores", "Income Stability", "Credit Utilisation", "Payment Behaviour", "Alt Data"]
            ext_val    = np.mean([ext_score_1, ext_score_2, ext_score_3])
            values = [
                ext_val,
                min(employment_yrs / 20, 1.0),
                max(0, 1 - credit_amount / (income + 1) / 3),
                ext_val * 0.9,
                min(doc_count / 10, 1.0),
            ]
            
            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.2,)}",
                line=dict(color=color, width=2),
                name="Applicant"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                height=280, margin=dict(t=30, b=10),
                title="Risk Factor Radar"
            )
            c2.plotly_chart(fig_radar, use_container_width=True)
        
        with tab2:
            if model_loaded and models:
                st.markdown("#### SHAP Feature Attribution")
                st.info("SHAP values show how each feature pushes the default probability up ↑ or down ↓")
                
                sv, base_val, df_aligned = get_shap_values(models[0], df_feat.copy(), feature_cols)
                
                # Sort by absolute SHAP
                shap_df = pd.DataFrame({
                    "Feature": df_aligned.columns,
                    "SHAP":    sv[0],
                    "Value":   df_aligned.iloc[0].values,
                }).sort_values("SHAP", key=abs, ascending=False).head(15)
                
                colors = ["#F44336" if v > 0 else "#4CAF50" for v in shap_df["SHAP"]]
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["SHAP"], y=shap_df["Feature"],
                    orientation="h", marker_color=colors,
                    text=[f"{v:+.4f}" for v in shap_df["SHAP"]], textposition="outside",
                ))
                fig_shap.update_layout(
                    title="Top 15 SHAP Feature Contributions (Red = Increases Risk, Green = Decreases)",
                    xaxis_title="SHAP Value", height=500,
                    margin=dict(l=150)
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.warning("⚠️ Load trained models to see SHAP explanations.")
                # Show mock
                mock_features = ["EXT_SOURCE_MEAN","CREDIT_INCOME_RATIO","AGE_YEARS","EMPLOYMENT_YEARS","BUREAU_COUNT",
                                 "EXT_SOURCE_3","NLP_EMB_0","ANNUITY_INCOME_RATIO","EXT_SOURCE_1","TOTAL_ENQUIRIES"]
                mock_shap     = np.array([-0.35, 0.28, -0.18, -0.12, 0.09, -0.22, -0.08, 0.15, -0.11, 0.06])
                colors = ["#F44336" if v > 0 else "#4CAF50" for v in mock_shap]
                fig_mock = go.Figure(go.Bar(
                    x=mock_shap, y=mock_features, orientation="h",
                    marker_color=colors, text=[f"{v:+.3f}" for v in mock_shap], textposition="outside"
                ))
                fig_mock.update_layout(title="Demo SHAP (load models for real values)", height=400, margin=dict(l=200))
                st.plotly_chart(fig_mock, use_container_width=True)
        
        with tab3:
            st.markdown("#### Concept Drift Sensitivity Analysis")
            st.markdown("How does this applicant's risk score change under economic shocks?")
            
            income_mults = np.linspace(0.2, 1.0, 9)
            drift_scores = []
            for mult in income_mults:
                drift_inp = dict(inputs)
                drift_inp["income"] = inputs["income"] * mult
                df_d = build_single_applicant_features(drift_inp, pca, sbert)
                if model_loaded:
                    s = predict_risk(df_d, models, feature_cols)
                else:
                    s = float(np.clip(risk_score + (1-mult)*0.25, 0, 0.99))
                drift_scores.append(s)
            
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=income_mults*100, y=[s*100 for s in drift_scores],
                mode="lines+markers", name="Default Probability",
                line=dict(color="#F44336", width=2.5),
                marker=dict(size=8, color=[
                    "#4CAF50" if s < 0.15 else "#FF9800" if s < 0.4 else "#F44336"
                    for s in drift_scores
                ])
            ))
            fig_drift.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
            fig_drift.add_hline(y=15, line_dash="dash", line_color="green",  annotation_text="Low Risk Threshold")
            fig_drift.update_layout(
                title="Default Probability vs Income Shock Severity",
                xaxis_title="Remaining Income (%)", yaxis_title="Default Probability (%)",
                height=400
            )
            st.plotly_chart(fig_drift, use_container_width=True)
        
        with tab4:
            st.markdown("#### Applicant Feature Summary")
            profile_data = {
                "Feature":       ["Annual Income", "Requested Credit", "Credit/Income Ratio", "Age",
                                  "Employment Years", "Ext Score (Mean)", "Alt Credit Score", "Family Size"],
                "Value":         [f"₹{income:,.0f}", f"₹{credit_amount:,.0f}",
                                  f"{credit_amount/max(income,1):.2f}x", f"{age} yrs",
                                  f"{employment_yrs} yrs", f"{np.mean([ext_score_1,ext_score_2,ext_score_3]):.3f}",
                                  f"{credit_score}/850", f"{family_size} members"],
                "Status":        ["✅" if income > 200000 else "⚠️",
                                  "✅" if credit_amount < income*3 else "⚠️",
                                  "✅" if credit_amount/max(income,1) < 2.5 else "🚨",
                                  "✅", "✅" if employment_yrs > 2 else "⚠️",
                                  "✅" if np.mean([ext_score_1,ext_score_2,ext_score_3]) > 0.5 else "🚨",
                                  "✅" if credit_score > 650 else "⚠️", "✅"],
            }
            st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center; font-size:0.8rem; color:#999;'>"
        "Built with LightGBM + XGBoost + Sentence-BERT + SHAP + River (ADWIN) + W&B | "
        "Home Credit Default Risk Dataset | "
        "For the 1.7B credit-invisible 🌍"
        "</p>", unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()