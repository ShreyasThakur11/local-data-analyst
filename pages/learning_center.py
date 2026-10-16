"""
Learning Center page — interactive knowledge base, statistical playbooks, and methodology reference.
Provides guidance on MBA analytics, financial modeling, operations, and statistical tests.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h2 style="margin:0; color:#38bdf8;">🎓 Analytics Learning Center & Methodology Playbooks</h2>
            <p style="margin:0.2rem 0 0; color:#94a3b8; font-size:0.95rem;">
                Rigorous reference library for statistical methods, domain business formulas, and machine learning principles.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Statistical Foundations",
        "💼 Domain Analytics Playbooks",
        "🧮 Interactive Calculators",
        "🤖 Machine Learning Primer",
    ])

    with tab1:
        st.subheader("Statistical Concepts & Interpretation")

        with st.expander("📊 1. Distributions, Skewness & Kurtosis", expanded=True):
            st.markdown(
                """
                - **Normal Distribution**: Symmetrical bell-shaped curve where mean = median = mode.
                - **Skewness**:
                  - *Skewness = 0*: Symmetric.
                  - *Positive Skew (> 0.5)*: Tail stretches to the right; mean > median. Frequent in income, customer spend, sales figures.
                  - *Negative Skew (< -0.5)*: Tail stretches to the left; mean < median. Common in manufacturing yields, age of retirement.
                - **Kurtosis**:
                  - *Mesokurtic (= 0 excess)*: Standard normal tails.
                  - *Leptokurtic (> 0)*: Heavy tails and sharp peak; higher risk of extreme outliers ("fat tails").
                  - *Platykurtic (< 0)*: Light tails and flat peak; fewer extreme deviations.
                """
            )

        with st.expander("🎯 2. Hypothesis Testing & P-Values"):
            st.markdown(
                """
                - **Null Hypothesis ($H_0$)**: Default assumption that no effect, difference, or relationship exists.
                - **Alternative Hypothesis ($H_1$)**: Claim that an effect or difference exists.
                - **P-Value**: The probability of observing a test statistic as extreme as or more extreme than the observed data, assuming $H_0$ is true.
                  - $p < 0.05$: Reject $H_0$ at the 5% significance level (statistically significant).
                  - $p \ge 0.05$: Fail to reject $H_0$ (insufficient evidence of an effect).
                - **Type I Error ($\alpha$)**: False positive (rejecting a true null).
                - **Type II Error ($\beta$)**: False negative (failing to reject a false null).
                - **Statistical Power ($1 - \beta$)**: Probability of detecting a true effect when one exists (typically targeted at $\ge 80\%$).
                """
            )

        with st.expander("🔗 3. Correlation vs. Causation & Regression"):
            st.markdown(
                """
                - **Pearson Correlation ($r$)**: Measures linear co-movement between two continuous variables ($[-1, 1]$).
                - **Spearman Rank Correlation ($\rho$)**: Measures monotonic relationships without assuming linearity or normality.
                - **Coefficient of Determination ($R^2$)**: Proportion of variance in the dependent variable explained by independent variables.
                - **Key Regression Assumptions (LINE)**:
                  1. **L**inearity: Relationship between predictors and target is linear.
                  2. **I**ndependence: Observations are independent of each other (Durbin-Watson test for autocorrelation).
                  3. **N**ormality: Residuals are approximately normally distributed (Q-Q plot, Shapiro-Wilk test).
                  4. **E**qual Variance (Homoscedasticity): Variance of residuals is constant across fitted values (Breusch-Pagan test).
                """
            )

    with tab2:
        st.subheader("Domain-Specific Analytics Playbooks")

        domain = st.selectbox(
            "Select Industry / Domain Framework",
            [
                "MBA Strategy & Growth Analytics",
                "Finance & Investment Analysis",
                "Operations & Supply Chain Management",
                "Sustainability & ESG Analytics",
            ],
        )

        if domain == "MBA Strategy & Growth Analytics":
            st.markdown(
                """
                ### Core Growth & Customer Metrics
                - **Customer Acquisition Cost (CAC)**:
                  $$\\text{CAC} = \\frac{\\text{Total Sales & Marketing Spend}}{\\text{Number of New Customers Acquired}}$$
                - **Customer Lifetime Value (CLV / LTV)**:
                  $$\\text{LTV} = \\frac{\\text{Average Order Value} \\times \\text{Purchase Frequency}}{\\text{Churn Rate}} \\times \\text{Gross Margin \\%}$$
                - **LTV / CAC Ratio Target**: Healthy SaaS/subscription businesses require $\\ge 3.0\\times$. Less than $1.0\\times$ destroys value.
                - **Net Promoter Score (NPS)**: $\\% \\text{Promoters (9-10)} - \\% \\text{Detractors (0-6)}$.
                """
            )

        elif domain == "Finance & Investment Analysis":
            st.markdown(
                """
                ### Financial Ratios & DuPont Analysis
                - **DuPont Analysis Framework**:
                  $$\\text{ROE} = \\underbrace{\\frac{\\text{Net Income}}{\\text{Revenue}}}_{\\text{Net Margin}} \\times \\underbrace{\\frac{\\text{Revenue}}{\\text{Assets}}}_{\\text{Asset Turnover}} \\times \\underbrace{\\frac{\\text{Assets}}{\\text{Equity}}}_{\\text{Equity Multiplier}}$$
                - **Operating Leverage**: Degree of Operating Leverage $(\\text{DOL}) = \\frac{\\% \\Delta \\text{EBIT}}{\\% \\Delta \\text{Sales}}$.
                - **Cash Conversion Cycle (CCC)**:
                  $$\\text{CCC} = \\text{Days Inventory Outstanding (DIO)} + \\text{Days Sales Outstanding (DSO)} - \\text{Days Payables Outstanding (DPO)}$$
                """
            )

        elif domain == "Operations & Supply Chain Management":
            st.markdown(
                """
                ### Supply Chain Operations & Inventory Control
                - **Economic Order Quantity (EOQ)**: Minimizes total holding and ordering costs.
                  $$\\text{EOQ} = \\sqrt{\\frac{2 \\times D \\times S}{H}}$$
                  *(where $D$ = Annual Demand, $S$ = Setup/Order Cost, $H$ = Annual Holding Cost per unit)*
                - **Safety Stock ($SS$)**:
                  $$SS = Z \\times \\sqrt{L \\times \\sigma_D^2 + D^2 \\times \\sigma_L^2}$$
                  *(where $Z$ = service factor, $L$ = lead time, $\\sigma_D$ = demand std dev, $\\sigma_L$ = lead time std dev)*
                - **Overall Equipment Effectiveness (OEE)**:
                  $$\\text{OEE} = \\text{Availability} \\times \\text{Performance} \\times \\text{Quality}$$
                """
            )

        elif domain == "Sustainability & ESG Analytics":
            st.markdown(
                """
                ### Carbon Accounting & GHG Protocols
                - **Scope 1**: Direct greenhouse gas emissions from owned or controlled sources (e.g., company boilers, fleet vehicles).
                - **Scope 2**: Indirect GHG emissions from the generation of purchased electricity, steam, heating, and cooling.
                - **Scope 3**: All other indirect emissions occurring in the upstream and downstream value chain.
                - **Carbon Intensity Ratio**: $\\frac{\\text{Total CO}_2\\text{e (metric tons)}}{\\text{Revenue (\\$M)}}$.
                """
            )

    with tab3:
        st.subheader("Interactive Formula Calculators")

        calc_choice = st.radio(
            "Select Tool",
            ["Economic Order Quantity (EOQ)", "Customer Lifetime Value (LTV)", "Sample Size for Proportions"],
            horizontal=True,
        )

        if calc_choice == "Economic Order Quantity (EOQ)":
            c1, c2, c3 = st.columns(3)
            with c1:
                annual_d = st.number_input("Annual Demand (units)", value=10000, step=500)
            with c2:
                order_cost = st.number_input("Cost per Order ($)", value=50.0, step=5.0)
            with c3:
                holding_cost = st.number_input("Annual Holding Cost per Unit ($)", value=2.5, step=0.5)

            if holding_cost > 0:
                eoq_val = math.sqrt((2 * annual_d * order_cost) / holding_cost)
                orders_per_yr = annual_d / eoq_val
                st.success(f"**Optimal Order Quantity (EOQ):** {eoq_val:,.1f} units")
                st.info(f"**Expected Orders per Year:** {orders_per_yr:,.1f} orders")

        elif calc_choice == "Customer Lifetime Value (LTV)":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                aov = st.number_input("Average Order Value ($)", value=120.0, step=10.0)
            with c2:
                pf = st.number_input("Purchase Frequency (orders/yr)", value=4.0, step=0.5)
            with c3:
                margin = st.slider("Gross Margin (%)", min_value=5, max_value=95, value=65) / 100.0
            with c4:
                churn = st.slider("Annual Churn Rate (%)", min_value=1, max_value=80, value=15) / 100.0

            if churn > 0:
                ltv_val = (aov * pf / churn) * margin
                st.success(f"**Customer Lifetime Value (LTV):** ${ltv_val:,.2f}")

        elif calc_choice == "Sample Size for Proportions":
            c1, c2, c3 = st.columns(3)
            with c1:
                cl = st.selectbox("Confidence Level", [90, 95, 99], index=1)
                z_map = {90: 1.645, 95: 1.96, 99: 2.576}
                z = z_map[cl]
            with c2:
                moe = st.slider("Margin of Error (%)", min_value=1, max_value=10, value=5) / 100.0
            with c3:
                p_est = st.slider("Estimated Baseline Proportion (%)", min_value=1, max_value=99, value=50) / 100.0

            if moe > 0:
                n_sample = (z ** 2 * p_est * (1 - p_est)) / (moe ** 2)
                st.success(f"**Recommended Sample Size:** {math.ceil(n_sample):,} participants")

    with tab4:
        st.subheader("Machine Learning Foundations for Analysts")
        st.markdown(
            """
            ### Supervised vs. Unsupervised Learning Overview

            | Method | Class | Primary Objective | Key Assumptions / Gotchas |
            | :--- | :--- | :--- | :--- |
            | **Linear Regression** | Supervised | Continuous value prediction & coefficient interpretation | Sensitivity to outliers, collinearity |
            | **Logistic Regression** | Supervised | Binary probability estimation ($p \\in [0, 1]$) | Requires linear log-odds relationship |
            | **K-Means Clustering** | Unsupervised | Segment data into $K$ spherical clusters | Must scale features; sensitive to $K$ choice |
            | **Isolation Forest** | Unsupervised | Anomaly and outlier score computation | Trees isolate rare points faster |
            | **Random Forest** | Supervised | Ensemble decision trees for classification/regression | Less prone to overfitting, provides feature importances |
            """
        )


if __name__ == "__main__":
    render()
