"""Advanced Analytics page — regression, clustering, outlier detection, forecasting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import histogram, regression_plot, scatter_plot
from components.metric_cards import active_dataset_guard, section_header
from core.statistics_engine import simple_regression

st.title("🧪 Advanced Analytics")
st.caption("Advanced methods that require sufficient data and validated assumptions. Not all methods apply to every dataset.")

rec = active_dataset_guard()
if rec is None:
    st.stop()

df = rec["df"]
profile = rec["profile"]
nc = profile.numeric_cols
cc = profile.categorical_cols

tab_reg, tab_cluster, tab_anomaly, tab_feature = st.tabs([
    "📐 Regression", "🎯 Clustering", "🚨 Anomaly Detection", "📊 Feature Importance"
])

# ─────────────────────────────────────────────────────────────────────────────
# Regression
# ─────────────────────────────────────────────────────────────────────────────
with tab_reg:
    section_header("Regression Analysis")

    if len(nc) < 2:
        st.warning("Need at least 2 numeric columns for regression.")
        st.stop()

    reg_type = st.radio("Regression type", ["Simple Linear (OLS)", "Multiple Linear (OLS)"], horizontal=True, key="reg_type")
    outcome = st.selectbox("Outcome (Y) variable", nc, key="reg_outcome")
    
    if reg_type == "Simple Linear (OLS)":
        predictor = st.selectbox("Predictor (X) variable", [c for c in nc if c != outcome], key="reg_predictor")

        if st.button("▶ Run Regression", key="btn_simple_reg", type="primary"):
            with st.spinner("Running OLS regression…"):
                result = simple_regression(df, predictor, outcome)

            if "error" in result:
                st.error(result["error"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R²", f"{result['r_squared']:.4f}")
                c2.metric("Slope", f"{result['slope']:.4f}")
                c3.metric("Intercept", f"{result['intercept']:.4f}")
                c4.metric("p-value", f"{result['p_value']:.4f}")

                st.markdown(f"**Equation:** `{result['equation']}`")
                st.caption(f"⚠️ {result['caveat']}")
                st.plotly_chart(
                    regression_plot(df, predictor, outcome, result["slope"],
                                   result["intercept"], result["r_squared"]),
                    use_container_width=True
                )

                # Residuals
                clean = df[[predictor, outcome]].dropna()
                y_pred = result["slope"] * clean[predictor] + result["intercept"]
                residuals = clean[outcome] - y_pred
                st.plotly_chart(
                    histogram(residuals, "Residuals", title="Residual Distribution"),
                    use_container_width=True
                )
                if abs(residuals.skew()) > 1:
                    st.warning("Residuals appear skewed — OLS assumptions may be violated.")

    else:  # Multiple
        predictors = st.multiselect("Predictor (X) variables", [c for c in nc if c != outcome], key="reg_multi_x")

        if predictors and st.button("▶ Run Multiple Regression", key="btn_multi_reg", type="primary"):
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_squared_error

                clean = df[predictors + [outcome]].dropna()
                n = len(clean)

                if n < len(predictors) + 5:
                    st.error(f"Too few observations ({n}) for {len(predictors)} predictors.")
                else:
                    X = clean[predictors].values
                    y = clean[outcome].values

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression().fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² (test set)", f"{r2:.4f}")
                    c2.metric("RMSE (test set)", f"{rmse:.4g}")
                    c3.metric("Train observations", f"{len(X_train):,}")

                    coef_df = pd.DataFrame({
                        "Predictor": predictors,
                        "Coefficient": model.coef_.round(6),
                    }).sort_values("Coefficient", key=abs, ascending=False)
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

                    st.caption(
                        "⚠️ Coefficients assume all predictors are in their original units. "
                        "R² on an 80/20 random split — not a substitute for proper cross-validation. "
                        "No causal interpretation."
                    )

            except ImportError:
                st.error("scikit-learn is required for multiple regression.")

# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────
with tab_cluster:
    section_header("K-Means Clustering (Exploratory)")
    st.info("Clustering is exploratory segmentation — not a predictive model. Cluster labels have no inherent order.")

    if len(nc) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
        st.stop()

    cluster_cols = st.multiselect("Select columns for clustering (2–8 recommended)", nc, default=nc[:min(4, len(nc))], key="cluster_cols")
    n_clusters = st.slider("Number of clusters (k)", 2, 10, 3, key="cluster_k")

    if len(cluster_cols) >= 2 and st.button("▶ Run K-Means", key="btn_kmeans", type="primary"):
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            clean = df[cluster_cols].dropna()
            if len(clean) < n_clusters * 5:
                st.error("Too few observations for reliable clustering.")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(clean)

                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)

                clean = clean.copy()
                clean["Cluster"] = [f"Cluster {l+1}" for l in labels]

                # Cluster summary
                summary = clean.groupby("Cluster")[cluster_cols].mean().round(2)
                st.markdown("**Cluster centroids (in original units):**")
                st.dataframe(summary, use_container_width=True)

                # Visualise first 2 dims
                if len(cluster_cols) >= 2:
                    fig = scatter_plot(
                        clean, cluster_cols[0], cluster_cols[1],
                        color_col="Cluster", add_trendline=False,
                        title=f"Cluster visualisation: {cluster_cols[0]} vs {cluster_cols[1]}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "⚠️ K-Means requires specifying k in advance, assumes spherical clusters, "
                    "and is sensitive to scale (pre-scaled here). "
                    "Clusters should be interpreted as starting points for exploration."
                )

        except ImportError:
            st.error("scikit-learn is required for clustering.")

# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────
with tab_anomaly:
    section_header("Anomaly Detection")
    st.info("Flags statistical outliers using Isolation Forest. These are unusual observations, not confirmed errors.")

    if not nc:
        st.warning("No numeric columns available.")
        st.stop()

    anomaly_cols = st.multiselect("Select columns", nc, default=nc[:min(4, len(nc))], key="anomaly_cols")
    contamination = st.slider("Expected anomaly rate (%)", 1, 20, 5, key="anomaly_contam") / 100

    if anomaly_cols and st.button("▶ Detect Anomalies", key="btn_anomaly", type="primary"):
        try:
            from sklearn.ensemble import IsolationForest

            clean = df[anomaly_cols].dropna()
            if len(clean) < 20:
                st.error("Need at least 20 observations for anomaly detection.")
            else:
                iso = IsolationForest(contamination=contamination, random_state=42)
                preds = iso.fit_predict(clean)
                clean = clean.copy()
                clean["Anomaly"] = preds == -1

                n_anomalies = int(clean["Anomaly"].sum())
                st.metric("Anomalies detected", f"{n_anomalies} ({n_anomalies/len(clean)*100:.1f}%)")

                anomaly_df = df.loc[clean[clean["Anomaly"]].index][anomaly_cols]
                st.markdown(f"**Top anomalous records ({min(20, n_anomalies)} shown):**")
                st.dataframe(anomaly_df.head(20), use_container_width=True)

                st.caption(
                    "⚠️ Isolation Forest is sensitive to the contamination rate parameter. "
                    "Results differ between runs when contamination does not match the true rate."
                )

        except ImportError:
            st.error("scikit-learn is required for anomaly detection.")

# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tab_feature:
    section_header("Feature Importance (Random Forest)")
    st.info(
        "Shows which columns are most predictive of the target variable in a Random Forest model. "
        "Importance does **not** imply causation."
    )

    if len(nc) < 2:
        st.warning("Need at least 2 numeric columns.")
        st.stop()

    target_col = st.selectbox("Target variable", nc, key="fi_target")
    feature_cols = st.multiselect("Feature columns", [c for c in nc if c != target_col], key="fi_features", default=[c for c in nc if c != target_col][:5])

    if feature_cols and st.button("▶ Compute Feature Importance", key="btn_fi", type="primary"):
        try:
            from sklearn.ensemble import RandomForestRegressor

            clean = df[feature_cols + [target_col]].dropna()
            if len(clean) < 20:
                st.error("Need at least 20 observations.")
            else:
                X = clean[feature_cols].values
                y = clean[target_col].values

                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)

                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": rf.feature_importances_,
                }).sort_values("Importance", ascending=False)

                st.dataframe(importance_df.round(4), use_container_width=True, hide_index=True)

                import plotly.express as px
                fig = px.bar(
                    importance_df, x="Importance", y="Feature", orientation="h",
                    title=f"Feature Importance for predicting {target_col}",
                    color_discrete_sequence=["#38bdf8"],
                )
                from components.charts import BASE_LAYOUT
                fig.update_layout(**BASE_LAYOUT, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "⚠️ Importance is measured within the Random Forest model. "
                    "Do not interpret as causal importance. "
                    "No train/test split applied here — use for exploration only."
                )
        except ImportError:
            st.error("scikit-learn is required for feature importance.")
