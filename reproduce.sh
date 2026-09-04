#!/bin/bash
# reproduce.sh
# Runs notebooks under Code/ for the DCEGM Search / durable-consumption project.
# Usage (from the repository root): bash reproduce.sh
# Requirements: conda environment 'DCEGM_Search_env' must be active
#   (see environment.yml).

set -e   # exit immediately if any notebook fails

NB_DIR="Code"

# Run a notebook in-place from Code/ (paths like ../Output, ../Figures expect that cwd)
run_nb() {
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    (
        cd "$NB_DIR" || exit 1
        jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=86400 \
            "$1"
    )
}

# ──────────────────────────────────────────────
# 1. FIGURES (retirement / search model)
# ──────────────────────────────────────────────

run_nb "Figures.ipynb"
run_nb "Figures_IncomeDiscMethod.ipynb"

# ──────────────────────────────────────────────
# 2. ONE-PERIOD DURABLE NOTEBOOKS
#    Order: baseline, taste, search, price, combined
# ──────────────────────────────────────────────

run_nb "DurableModel_One_Period.ipynb"
run_nb "DurableModel_TasteShock_One_Period.ipynb"
run_nb "DurableModel_Search_One_Period.ipynb"
run_nb "DurableModel_Price_One_Period.ipynb"
run_nb "DurableModel_Combined_One_Period.ipynb"

# ──────────────────────────────────────────────
# 3. FIGURES — DURABLE MODEL
# ──────────────────────────────────────────────

run_nb "Figures_Durable.ipynb"

# ──────────────────────────────────────────────
# 4. INFINITE HORIZON
#    Channel notebooks first (Taste, Search, Price), then combined Inf results.
# ──────────────────────────────────────────────

run_nb "DurableModel_Taste_Inf.ipynb"
run_nb "DurableModel_Search_Inf.ipynb"
run_nb "DurableModel_Price_Inf.ipynb"
run_nb "DurableModel_Combined_Inf.ipynb"
run_nb "DurableModel_Inf.ipynb"

echo ""
echo "=========================================="
echo "All notebooks completed successfully."
echo "=========================================="
