"""AIRLINE SATISFACTION INTELLIGENCE
Streamlit Web Application for End-to-End Passenger Satisfaction Prediction and Insights.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ARTIFACTS_DIR,
    CATBOOST_MODEL_PATH,
    DATA_DIR,
    ENSEMBLE_CONFIG_PATH,
    EVALUATION_METRICS_PATH,
    FEATURE_SCHEMA_PATH,
    FINAL_CAT_WEIGHT,
    FINAL_LGB_WEIGHT,
    LGBM_MODEL_PATH,
    MODEL_COMPARISON_PATH,
    MODELS_DIR,
    PREPROCESSOR_ARTIFACTS_PATH,
    RATING_COLS,
    REPORTS_DIR,
)
from src.model_loader import are_artifacts_available, load_model_bundle
from src.predict import predict_passengers, predict_single_passenger
from src.preprocessing import RAW_FEATURE_COLUMNS, validate_schema

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Airline Satisfaction Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN DARK THEME CSS ---
st.markdown("""
<style>
    /* Dark Theme Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    
    /* Header card */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Metrics and stat cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Prediction Banner */
    .prediction-satisfied {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #ecfdf5;
    }
    
    .prediction-dissatisfied {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.25) 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #fef2f2;
    }
    
    /* Warning setup box */
    .setup-box {
        background-color: #1e1e2f;
        border-left: 4px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 6px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model_bundle():
    """Caches loaded model bundle in memory."""
    ready, status = are_artifacts_available()
    if not ready:
        return None
    try:
        return load_model_bundle()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


bundle = get_model_bundle()
artifacts_ready, artifact_status = are_artifacts_available()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("## ✈️ Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🛫 Overview",
        "🔮 Passenger Prediction",
        "📂 Batch Prediction",
        "📊 Model Performance",
        "🔍 Feature Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
if artifacts_ready:
    st.sidebar.success("✅ Models Active & Loaded")
    if bundle and bundle.has_catboost:
        st.sidebar.info("Ensemble: LightGBM (47.5%) + CatBoost (52.5%)")
    else:
        st.sidebar.info("Model: LightGBM (Primary)")
else:
    st.sidebar.warning("⚠️ Model Artifacts Not Yet Generated")
    st.sidebar.caption("Run training pipeline to generate artifacts:")
    st.sidebar.code("python -m src.train", language="bash")

st.sidebar.markdown("---")
st.sidebar.caption("Kaggle Playground Series s6e10")
st.sidebar.caption("Benchmark Public LB: **0.96032**")


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "🛫 Overview":
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">AIRLINE SATISFACTION INTELLIGENCE</h1>
        <div class="header-subtitle">Enterprise-Grade Machine Learning Classification & Operational Decision Support System</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.96032</div>
            <div class="metric-label">Kaggle Benchmark (Public LB)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">ROC-AUC</div>
            <div class="metric-label">Primary Metric</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">LGBM + CatBoost</div>
            <div class="metric-label">Ensemble Architecture</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">13 Indicators</div>
            <div class="metric-label">Passenger Survey Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Executive Summary")
    st.write("""
    **Airline Satisfaction Intelligence** is an end-to-end, production-oriented machine learning solution designed to identify and predict airline passenger satisfaction.
    Originating from the Kaggle *Predicting Airline Satisfaction (Playground Series Season 6 Episode 10)* competition, this project transforms experimental notebooks into an enterprise-ready modular architecture featuring:
    """)

    st.markdown("""
    - **Leakage-Safe Preprocessing**: Rigorous out-of-fold smoothed target encoding on route distance proxies, ensuring zero validation or test leakage.
    - **Engineered Survey Indices**: Domain-specific rating aggregations capturing Comfort (`Seat comfort`, `Leg room`, `Inflight entertainment`), Convenience (`Booking`, `Boarding`, `Flight times`), and Service (`On-board`, `Baggage`, `Check-in`, `Cleanliness`).
    - **Dual Gradient-Boosted Ensemble**: Combines LightGBM (47.5% weight) with CatBoost (52.5% weight), capturing both one-hot structural representations and high-cardinality categorical splits.
    - **Operational Decision Interface**: Real-time interactive passenger inference, batch CSV ingestion, validation telemetry, and feature importance explanations.
    """)

    st.markdown("### 🏗️ Pipeline Architecture")
    st.markdown("""
    ```
    Raw Passenger Inputs (Demographics, Route, Delays, 13 Survey Ratings)
                             │
                             ▼
    [Phase 3: Preprocessing & Leakage-Safe Feature Engineering]
      ├── Total Delay & Delay Differences
      ├── Log Distance & Smoothed OOF Distance Target Encoding
      ├── Rating Distributions (Mean, Min, Max, Std, Extremes)
      └── Interaction Ratios (Comfort x Service, Convenience x Comfort, Age x Distance)
                             │
                             ▼
    [Phase 4: Dual-Model Gradient Boosting Validation]
      ├── LightGBM (num_leaves=127, lr=0.025, colsample=0.85)
      └── CatBoost (depth=8, lr=0.04, l2_leaf_reg=5)
                             │
                             ▼
    [Phase 5: Weighted Probability Ensemble]
      Final Probability = 0.475 * P(LightGBM) + 0.525 * P(CatBoost)
                             │
                             ▼
    [Phase 6: Production Inference & Operational Streamlit UI]
    ```
    """)


# ==============================================================================
# PAGE 2: PASSENGER PREDICTION
# ==============================================================================
elif page == "🔮 Passenger Prediction":
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">Individual Passenger Prediction</h1>
        <div class="header-subtitle">Evaluate satisfaction likelihood for a single passenger profile in real-time</div>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts_ready:
        st.markdown("""
        <div class="setup-box">
            <strong>⚠️ Model Artifacts Missing</strong><br>
            Trained model artifacts were not detected in <code>models/</code> and <code>artifacts/</code>.<br>
            Run the training script to generate the models and preprocessor before running live inference:<br>
            <code>python -m src.train</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.form("passenger_form"):
            st.markdown("#### 1. Passenger Demographics & Flight Details")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
            with c2:
                age = st.slider("Age", 7, 85, 38)
                travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            with c3:
                flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
                flight_distance = st.number_input("Flight Distance (miles)", min_value=30, max_value=6000, value=1200, step=50)
            with c4:
                dep_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=1500, value=0, step=5)
                arr_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=1500, value=0, step=5)

            st.markdown("---")
            st.markdown("#### 2. Passenger Experience Ratings (0: N/A, 1: Very Poor → 5: Exceptional)")

            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.caption("💺 Comfort & Environment")
                seat_comfort = st.slider("Seat comfort", 0, 5, 4)
                leg_room = st.slider("Leg room service", 0, 5, 4)
                cleanliness = st.slider("Cleanliness", 0, 5, 4)
                food_drink = st.slider("Food and drink", 0, 5, 3)

            with r_col2:
                st.caption("💻 Digital & Booking Services")
                wifi = st.slider("Inflight wifi service", 0, 5, 4)
                online_boarding = st.slider("Online boarding", 0, 5, 4)
                online_booking = st.slider("Ease of Online booking", 0, 5, 3)
                time_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 4)
                gate_location = st.slider("Gate location", 0, 5, 3)

            with r_col3:
                st.caption("🧑‍✈️ Cabin & Ground Crew")
                inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 4)
                onboard_service = st.slider("On-board service", 0, 5, 4)
                baggage = st.slider("Baggage handling", 0, 5, 4)
                checkin = st.slider("Checkin service", 0, 5, 4)

            threshold = st.slider("Classification Decision Threshold", 0.1, 0.9, 0.5, 0.05)
            submit_btn = st.form_submit_button("⚡ Predict Satisfaction", use_container_width=True)

        if submit_btn:
            passenger_data = {
                "Gender": gender,
                "Customer Type": customer_type,
                "Age": age,
                "Type of Travel": travel_type,
                "Class": flight_class,
                "Flight Distance": flight_distance,
                "Departure Delay in Minutes": dep_delay,
                "Arrival Delay in Minutes": arr_delay,
                "Inflight wifi service": wifi,
                "Departure/Arrival time convenient": time_convenient,
                "Ease of Online booking": online_booking,
                "Gate location": gate_location,
                "Food and drink": food_drink,
                "Online boarding": online_boarding,
                "Seat comfort": seat_comfort,
                "Inflight entertainment": inflight_entertainment,
                "On-board service": onboard_service,
                "Leg room service": leg_room,
                "Baggage handling": baggage,
                "Checkin service": checkin,
                "Cleanliness": cleanliness,
            }

            res = predict_single_passenger(passenger_data, bundle=bundle, threshold=threshold)
            prob = res["satisfaction_probability"]
            is_satisfied = res["prediction"] == 1

            st.markdown("### 📊 Prediction Result")
            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                if is_satisfied:
                    st.markdown(f"""
                    <div class="prediction-satisfied">
                        <h2 style="margin:0; font-size:2rem;">🎉 SATISFIED</h2>
                        <p style="font-size:1.2rem; margin:10px 0 0 0;">Satisfaction Probability: <strong>{prob:.1%}</strong></p>
                        <p style="color:#6ee7b7; font-size:0.9rem;">Model indicates a high positive passenger experience likelihood.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-dissatisfied">
                        <h2 style="margin:0; font-size:2rem;">⚠️ NEUTRAL OR DISSATISFIED</h2>
                        <p style="font-size:1.2rem; margin:10px 0 0 0;">Satisfaction Probability: <strong>{prob:.1%}</strong></p>
                        <p style="color:#fca5a5; font-size:0.9rem;">Model flags risk of dissatisfaction. Proactive intervention recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(f"Inference Engine: {res['model_used']} | Threshold: {threshold}")
                st.info("ℹ️ Note: This is an estimated statistical model prediction for decision support, not an absolute guarantee.")

            with res_c2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Satisfaction Confidence (%)", 'font': {'color': '#f1f5f9'}},
                    number={'suffix': "%", 'font': {'color': '#38bdf8'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#38bdf8"},
                        'steps': [
                            {'range': [0, threshold * 100], 'color': "rgba(239, 68, 68, 0.3)"},
                            {'range': [threshold * 100, 100], 'color': "rgba(16, 185, 129, 0.3)"},
                        ],
                        'threshold': {
                            'line': {'color': "#ffffff", 'width': 3},
                            'thickness': 0.8,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="#1e293b",
                    font={'color': "#f1f5f9"},
                    height=260,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 3: BATCH PREDICTION
# ==============================================================================
elif page == "📂 Batch Prediction":
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">Batch Passenger Prediction</h1>
        <div class="header-subtitle">Upload a CSV file containing multiple passenger records to compute satisfaction scores in bulk</div>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts_ready:
        st.markdown("""
        <div class="setup-box">
            <strong>⚠️ Model Artifacts Missing</strong><br>
            Please train the models before running batch inference: <code>python -m src.train</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        sample_path = DATA_DIR / "sample_passengers.csv"
        c1, c2 = st.columns([2, 1])
        with c1:
            uploaded_file = st.file_uploader("Upload Passenger CSV file", type=["csv"])
        with c2:
            st.markdown("#### Need a test file?")
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Sample Passengers CSV",
                        data=f.read(),
                        file_name="sample_passengers.csv",
                        mime="text/csv",
                    )
            use_sample = st.button("🚀 Load Sample Dataset Directly")

        df_to_predict = None
        if uploaded_file is not None:
            df_to_predict = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file loaded successfully: {len(df_to_predict)} passenger records.")
        elif use_sample and sample_path.exists():
            df_to_predict = pd.read_csv(sample_path)
            st.info(f"Loaded built-in sample dataset: {len(df_to_predict)} passenger records.")

        if df_to_predict is not None:
            st.markdown("#### Input Data Preview")
            st.dataframe(df_to_predict.head(5), use_container_width=True)

            is_valid, missing = validate_schema(df_to_predict, is_training=False)
            if not is_valid:
                st.warning(f"Note: Some expected raw columns were missing and will use default imputation: {missing}")

            threshold_batch = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05, key="batch_thresh")
            if st.button("⚡ Generate Batch Predictions", type="primary"):
                with st.spinner("Processing passenger records through ensemble pipeline..."):
                    preds = predict_passengers(df_to_predict, bundle=bundle, threshold=threshold_batch)
                    combined_results = pd.concat([df_to_predict.reset_index(drop=True), preds], axis=1)

                st.markdown("### 📈 Prediction Results Summary")
                b1, b2, b3 = st.columns(3)
                total = len(preds)
                sat_count = (preds["prediction"] == 1).sum()
                dissat_count = total - sat_count

                with b1:
                    st.metric("Total Passengers Processed", f"{total:,}")
                with b2:
                    st.metric("Predicted Satisfied", f"{sat_count:,} ({sat_count/total:.1%})")
                with b3:
                    st.metric("Predicted Dissatisfied", f"{dissat_count:,} ({dissat_count/total:.1%})")

                # Distribution chart
                fig = px.histogram(
                    preds,
                    x="satisfaction_probability",
                    nbins=25,
                    title="Satisfaction Probability Distribution",
                    color="satisfaction_label",
                    color_discrete_map={
                        "Satisfied": "#10b981",
                        "Neutral or Dissatisfied": "#ef4444"
                    },
                    labels={"satisfaction_probability": "Predicted Satisfaction Probability"}
                )
                fig.update_layout(
                    paper_bgcolor="#1e293b",
                    plot_bgcolor="#1e293b",
                    font={'color': "#f1f5f9"}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Output Preview")
                st.dataframe(combined_results.head(15), use_container_width=True)

                csv_bytes = combined_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Full Predictions CSV",
                    data=csv_bytes,
                    file_name="passenger_satisfaction_predictions.csv",
                    mime="text/csv",
                )


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">Model Performance & Evaluation</h1>
        <div class="header-subtitle">Transparent validation metrics, ROC curves, confusion matrix, and Kaggle leaderboard benchmark</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("📌 **Evaluation Context**: The primary evaluation metric for this competition is **ROC-AUC**. The established Kaggle public leaderboard benchmark is approximately **0.96032**.")

    if MODEL_COMPARISON_PATH.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
        st.markdown("### 🏆 Validation Model Comparison Table")
        st.dataframe(comp_df, use_container_width=True)

        if EVALUATION_METRICS_PATH.exists():
            with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            # ROC Curves
            fig = go.Figure()
            for m in metrics_data:
                roc_pts = m.get("roc_curve", {})
                if roc_pts:
                    fig.add_trace(go.Scatter(
                        x=roc_pts["fpr"],
                        y=roc_pts["tpr"],
                        mode="lines",
                        name=f"{m['model_name']} (AUC = {m['roc_auc']:.4f})",
                        line=dict(width=2)
                    ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(dash="dash", color="#64748b")
            ))
            fig.update_layout(
                title="Validation ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                paper_bgcolor="#1e293b",
                plot_bgcolor="#1e293b",
                font={'color': "#f1f5f9"},
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confusion Matrix display
            st.markdown("### 🔲 Confusion Matrices")
            cm_cols = st.columns(len(metrics_data))
            for idx, m in enumerate(metrics_data):
                with cm_cols[idx]:
                    st.caption(f"**{m['model_name']}**")
                    cm = np.array(m["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Dissatisfied (0)", "Satisfied (1)"],
                        y=["Dissatisfied (0)", "Satisfied (1)"]
                    )
                    fig_cm.update_layout(
                        paper_bgcolor="#1e293b",
                        plot_bgcolor="#1e293b",
                        font={'color': "#f1f5f9"},
                        height=300
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.markdown("""
        <div class="setup-box">
            <strong>ℹ️ Evaluation Report Not Found</strong><br>
            A local validation comparison report has not been generated yet. Run the training script:<br>
            <code>python -m src.train</code><br>
            This will automatically compute and display the stratified validation ROC-AUC, accuracy, precision, recall, and ROC curves.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: FEATURE INSIGHTS
# ==============================================================================
elif page == "🔍 Feature Insights":
    st.markdown("""
    <div class="main-header">
        <h1 class="header-title">Feature Insights & Factor Analysis</h1>
        <div class="header-subtitle">Key satisfaction drivers, survey dimensions, and gradient-boosted feature importances</div>
    </div>
    """, unsafe_allow_html=True)

    if bundle and bundle.lgb_model:
        model = bundle.lgb_model
        if hasattr(model, "feature_importances_") and hasattr(model, "feature_name_"):
            feat_imp = pd.DataFrame({
                "Feature": model.feature_name_,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(20)

            st.markdown("### 🌲 LightGBM Top 20 Feature Importances (Split-based)")
            fig = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Top Predictive Features in Airline Passenger Satisfaction",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#1e293b",
                plot_bgcolor="#1e293b",
                font={'color': "#f1f5f9"},
                height=550,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="setup-box">
            <strong>ℹ️ Model Not Loaded</strong><br>
            Feature importance charts require the trained LightGBM model. Run <code>python -m src.train</code> to generate them.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💡 Core Domain Drivers")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **1. Digital & Boarding Ease**
        - Online boarding and Inflight Wi-Fi consistently rank among the strongest differentiators for passenger satisfaction.
        - Frictionless check-in and intuitive booking workflows offset modest flight delays.
        """)
    with c2:
        st.markdown("""
        **2. Comfort & Entertainment**
        - Seat comfort, Leg room service, and Inflight entertainment form the physical passenger experience core.
        - Business travel passenger satisfaction is particularly sensitive to seat comfort and workspace convenience.
        """)
    with c3:
        st.markdown("""
        **3. Delay Asymmetry**
        - Arrival delays generally penalize satisfaction more severely than departure delays.
        - Total delay interactions combined with distance capture route disruption risks.
        """)

