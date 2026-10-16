#!/usr/bin/env bash
# Quick launcher for Local Data Analyst
set -e

echo "🚀 Starting Local Data Analyst..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
