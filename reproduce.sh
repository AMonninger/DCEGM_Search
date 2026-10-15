#!/usr/bin/env bash
# reproduce.sh — figures in Text/dcegmSearch.tex
#
# Order (logs/code.txt):
#   1. Code/Figures_Retirement.ipynb
#   2. Code/Figures_Durable.ipynb
#   3. Infinite-horizon channel notebooks, then Code/DurableModel_Inf.ipynb
#   4. The two timeline pictures in Text/
#
# DEBUG and OLD notebooks are not run. Several of them overwrite a paper
# pickle or a paper PDF.
#
# EV_vs_SearchPhi.pdf is the saved comparison at sigma_zeta = 0.10
# (Output/Retirement_EV_vs_SearchPhi.pkl). This script does not
# regenerate those two files. Figures_Retirement.ipynb also writes
# EV_vs_SearchPhi_EV015.pdf, which is not in the text. That cell has
# optimize = False and reads Output/Retirement_EV_vs_SearchPhi_EV015.pkl.
#
# Environment: conda env from environment.yml (name: DCEGM_Search_env).
# Uses `conda run`. Run with bash, not sh.
#
# Usage:
#   bash reproduce.sh
#   NB_TIMEOUT=86400 bash reproduce.sh   # seconds per notebook; -1 is no limit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DCEGM_SEARCH_ROOT="${REPO_ROOT}"

ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="DCEGM_Search_env"
CODE_DIR="${REPO_ROOT}/Code"
TEXT_DIR="${REPO_ROOT}/Text"
NB_TIMEOUT="${NB_TIMEOUT:--1}"

find_conda() {
    if command -v conda &>/dev/null; then
        command -v conda
        return 0
    fi
    local user_name="${USERNAME:-${USER:-}}"
    local candidate
    for candidate in \
        "/opt/conda/bin/conda" \
        "${HOME}/anaconda3/bin/conda" \
        "${HOME}/miniconda3/bin/conda" \
        "${HOME}/AppData/Local/miniconda3/Scripts/conda.exe" \
        "${HOME}/AppData/Local/anaconda3/Scripts/conda.exe" \
        "/c/Users/${user_name}/anaconda3/Scripts/conda.exe" \
        "/c/Users/${user_name}/miniconda3/Scripts/conda.exe" \
        "/c/Users/${user_name}/AppData/Local/miniconda3/Scripts/conda.exe" \
        "/c/Users/${user_name}/AppData/Local/anaconda3/Scripts/conda.exe"
    do
        if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! CONDA_BIN="$(find_conda)"; then
    echo "ERROR: conda not found. Create the env from environment.yml." >&2
    exit 1
fi
CONDA=("${CONDA_BIN}")

conda_run() {
    "${CONDA[@]}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

run_nb() {
    local nb="$1"
    echo "==> [${ENV_NAME}] Code/${nb}"
    (
        cd "${CODE_DIR}"
        conda_run jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout="${NB_TIMEOUT}" \
            --ExecutePreprocessor.kernel_name="${ENV_NAME}" \
            "./${nb}"
    )
}

compile_timeline() {
    local texfile="$1"
    local base="${texfile%.tex}"
    echo "==> LaTeX: Text/${texfile}"
    (
        cd "${TEXT_DIR}"
        pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
            "${texfile}"
    )
    echo "    PDF -> ${TEXT_DIR}/${base}.pdf"
}

export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo ""
echo "=== 0. Conda environment ==="
if "${CONDA[@]}" run -n "${ENV_NAME}" true 2>/dev/null; then
    echo "Using existing env: ${ENV_NAME}"
else
    echo "Creating conda env '${ENV_NAME}' from ${ENV_FILE} ..."
    "${CONDA[@]}" env create -f "${ENV_FILE}"
fi
conda_run python -m ipykernel install --user --name "${ENV_NAME}" --display-name "${ENV_NAME}"
echo "Python: $(conda_run python -c 'import sys; print(sys.executable)')"

if ! command -v pdflatex &>/dev/null; then
    echo "ERROR: pdflatex not found. Install TeX Live or MiKTeX." >&2
    exit 1
fi

# ──────────────────────────────────────────────
echo ""
echo "=== 1. Retirement figures (Text/RetirementModel.tex) ==="
echo "    Writes Figures/Figures_v4/. Does not overwrite EV_vs_SearchPhi.pdf."
run_nb "Figures_Retirement.ipynb"

# ──────────────────────────────────────────────
echo ""
echo "=== 2. Durable finite horizon (Text/DurableModelBasic.tex) ==="
echo "    Writes Figures/DurableModel/."
run_nb "Figures_Durable.ipynb"

# ──────────────────────────────────────────────
echo ""
echo "=== 3. Durable infinite horizon ==="
echo "    Channel notebooks first. The assembler only reads their pickles."
echo "    Close any PDF in Figures/DurableModel that a viewer still has open."
run_nb "DurableModel_Taste_Inf.ipynb"
run_nb "DurableModel_Search_Inf.ipynb"
run_nb "DurableModel_Price_Inf.ipynb"
run_nb "DurableModel_Combined_varS_Inf.ipynb"
run_nb "DurableModel_Combined_varP_Inf.ipynb"
echo "    Assembler: steady-state table in the text. No Jacobian PDF."
run_nb "DurableModel_Inf.ipynb"

# ──────────────────────────────────────────────
echo ""
echo "=== 4. Timeline figures ==="
compile_timeline "timeline_RetirementModel.tex"
compile_timeline "timeline_DurableModel.tex"

printf '\nDone.\n'
printf '  Retirement figures: %s\n' "${REPO_ROOT}/Figures/Figures_v4"
printf '  Durable figures:    %s\n' "${REPO_ROOT}/Figures/DurableModel"
printf '  Steady-state table: %s\n' "${REPO_ROOT}/Output/DurableModel_Inf_SS_short.tex"
printf '  Timelines:          %s\n' "${TEXT_DIR}"
