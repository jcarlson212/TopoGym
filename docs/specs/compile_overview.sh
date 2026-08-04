#!/usr/bin/env bash
# Compile topo_gym_overview.tex -> topo_gym_overview.pdf
# Usage: ./compile_overview.sh
set -euo pipefail

cd "$(dirname "$0")"

JOB="topo_gym_overview"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error "${JOB}.tex"
  latexmk -c "${JOB}.tex" >/dev/null 2>&1 || true
else
  # Two passes so labels/cross-references resolve.
  pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex"
  pdflatex -interaction=nonstopmode -halt-on-error "${JOB}.tex"
  rm -f "${JOB}".{aux,log,out,toc,fls,fdb_latexmk}
fi

echo "Built ${JOB}.pdf"
