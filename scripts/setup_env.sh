#!/usr/bin/env bash
# Set up the SynCo environment: pinned Python packages + the bundled, patched UnityMAS-O.
#   bash scripts/setup_env.sh            # installs into the active Python environment
#   VENV=.venv bash scripts/setup_env.sh # creates / uses a virtualenv first
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

UMO_DIR="${ROOT}/third_party/UnityMAS-O"   # UnityMAS-O @ 7f1616a + SynCo changes (see SYNCO_MODIFICATIONS.md)

if [[ -n "${VENV:-}" ]]; then
  [[ -d "${VENV}" ]] || python3.10 -m venv "${VENV}"
  source "${VENV}/bin/activate"
fi
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt

# UnityMAS-O (verl fork with the star_ppo multi-agent trainer), bundled in third_party/.
cd "${UMO_DIR}"
python -m pip install -e . --no-deps
cd "${ROOT}"
python -m pip install -e . --no-deps

echo "[setup] done"
