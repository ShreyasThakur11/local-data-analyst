# 📊 Local Data Analyst

> **A local-first, personal analytics copilot and decision intelligence platform.**  
> Built for rigorous MBA coursework, enterprise consulting, and data engineering workflows without compromising data privacy.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Engine](https://img.shields.io/badge/DuckDB-In--Memory%20SQL-FFF000.svg)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)](tests/)

---

## 🌟 Key Highlights

- 🔒 **100% Local-First & Air-Gapped Ready**: All computations run locally on your machine. Zero dataset uploads, zero telemetry, zero unexpected cloud egress.
- ⚡ **High-Performance Analytics**: Dual-engine ingestion with **Pandas** and **DuckDB**, backed by Parquet caching for instant reloads.
- 🎯 **11 Tailored Analytical Profiles**: Custom KPIs, recommendations, and semantic roles for MBA Statistics, Finance, Marketing, Supply Chain, Operations, and ESG Sustainability.
- 🩺 **Non-Destructive Data Hygiene**: Comprehensive quality scorecards, anomaly flags, and safe preview-before-apply transformation logging.
- 🤖 **Analyst Assistant & SQL Studio**: Plain-English natural language queries mapped deterministically to DuckDB SQL, plus an interactive SQL sandbox with safety guardrails.
- 📄 **Multi-Format Publication Reports**: Export executive briefings and technical audits to standalone styled HTML, Markdown, and machine-readable JSON.
- 🧠 **Adaptive Learning Engine**: SQLite-backed feedback logging with Thompson Sampling multi-armed bandits to learn what analysis workflows you prefer.

---

## 🏗️ Architecture & Modules

```
local_data_analyst/
├── app.py                      # Master Streamlit application & sidebar router
├── pyproject.toml              # Project dependencies & build metadata
├── requirements.txt            # Pinned requirements
├── components/                 # Reusable UI & plotting primitives
│   ├── charts.py               # Dark-theme Plotly visualizations
│   └── metric_cards.py         # KPI cards, data dictionaries, feedback controls
├── core/                       # Core analytical computation engines
│   ├── ingestion.py            # CSV, Excel, Parquet file loaders
│   ├── type_inference.py       # Heuristic column type & semantic role inference
│   ├── profiling.py            # Summary statistics, quality scoring, missingness
│   ├── statistics_engine.py    # Correlations, ANOVA, Kruskal-Wallis, distributions
│   ├── data_quality.py         # Audit rules & non-destructive transformations
│   ├── query_engine.py         # DuckDB read-only SQL sandbox
│   ├── recommendation_engine.py# Context-aware analytical recommendations
│   ├── insights_engine.py      # Grounded plain-English interpretations
│   └── cache_manager.py        # Parquet-backed local file caching
├── learning/                   # Personalization & preference storage
│   └── preference_store.py     # SQLite persistence & Thompson sampling bandit
├── pages/                      # 13 Dedicated analytical pages
│   ├── workspace.py            # Dataset upload, local file selection & previews
│   ├── dataset_overview.py     # KPI scorecard & quality health summary
│   ├── column_explorer.py      # Deep-dive univariate inspections & histograms
│   ├── data_quality.py         # Audit findings & transformation workbench
│   ├── column_mapping.py       # Semantic roles & variable mapping
│   ├── relationship_analysis.py# Bivariate correlations, scatter, ANOVA, cross-tabs
│   ├── advanced_analysis.py    # Regressions, K-Means clustering, Isolation Forest
│   ├── dashboard_builder.py    # Interactive multi-panel dashboard studio
│   ├── consultant_mode.py      # Structured evidence-based insight generator
│   ├── analyst_assistant.py    # Natural language queries & DuckDB SQL console
│   ├── learning_center.py      # Statistical guides & domain playbooks
│   ├── report_export.py        # Standalone HTML, Markdown, JSON report generator
│   └── settings.py             # Cache controls, privacy audit, optional LLM keys
├── sample_data/                # Synthetic enterprise test datasets
│   ├── generate_samples.py     # Deterministic sample dataset generator
│   ├── sales_and_revenue.csv   # Retail / eCommerce transactions
│   ├── manufacturing_operations.csv # Equipment sensor & yield telemetry
│   └── energy_and_carbon.csv   # Facility Scope 1/2/3 emissions & intensity
├── tests/                      # Automated test suite
│   └── test_all_core.py        # Unit tests covering all core modules
└── utils/                      # Common configuration, logging, validation
    ├── configuration.py        # App settings & directory paths
    ├── logging_config.py       # Structured logging setup
    ├── security.py             # SQL safety checks & content hashes
    └── validation.py           # Path & identifier sanitization
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.9 or higher
- Git

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/ShreyasThakur11/local-data-analyst.git
cd local-data-analyst

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Sample Datasets (Optional)
Generate standard benchmark datasets with known properties:
```bash
python3 sample_data/generate_samples.py
```

### 4. Launch Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Validation

Run the automated test suite with `pytest`:
```bash
python3 -m pytest tests/ -v
```

All 9 core test suites validate:
- Column type detection & semantic role categorization
- High-speed statistical profiling & quality score calculation
- Parametric & non-parametric hypothesis tests (ANOVA, Kruskal-Wallis)
- Read-only DuckDB SQL safety validation
- Non-destructive transformation sandboxing & logging
- Parquet caching lifecycle
- Thompson sampling multi-armed bandit recommendation updates

---

## 🛡️ Security & Privacy Guarantees

1. **Local-First Execution**: Datasets remain entirely on the local device.
2. **DuckDB Sandboxing**: File writing and external network access inside SQL queries are disabled via `SET enable_external_access = FALSE`.
3. **Strict Read-Only Enforcement**: Query parser rejects any statements mutating database state (`DROP`, `INSERT`, `UPDATE`, `ALTER`, etc.).
4. **Data Immutability**: All data transformations create new versions and log steps without modifying raw input files.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shreyas Thakur**  
- MBA Candidate, Indian Institute of Management (IIM), Mumbai  
- GitHub: [@ShreyasThakur11](https://github.com/ShreyasThakur11)  
- Portfolio: [shreyasthakur11.github.io](https://shreyasthakur11.github.io)
