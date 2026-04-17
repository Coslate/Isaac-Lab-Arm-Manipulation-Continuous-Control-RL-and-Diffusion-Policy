#!/usr/bin/env bash
# Restore the `isaac_arm` conda env from requirement.txt plus the two editable
# installs of isaaclab's in-wheel sub-packages.
#
# Run inside an already-activated conda env (e.g. `conda activate isaac_arm`).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[install_isaac] pip install -r ${REPO_ROOT}/requirement.txt"
pip install -r "${REPO_ROOT}/requirement.txt"

ISAACLAB_DIR="$(python -c 'import isaaclab, os; print(os.path.dirname(isaaclab.__file__))')"
echo "[install_isaac] isaaclab wheel installed at: ${ISAACLAB_DIR}"

for subpkg in isaaclab_assets isaaclab_tasks; do
  subpkg_dir="${ISAACLAB_DIR}/source/${subpkg}"
  if [[ ! -d "${subpkg_dir}" ]]; then
    echo "[install_isaac] ERROR: expected sub-package dir missing: ${subpkg_dir}" >&2
    exit 1
  fi
  echo "[install_isaac] pip install -e ${subpkg_dir}"
  pip install -e "${subpkg_dir}"
done

echo "[install_isaac] done."
