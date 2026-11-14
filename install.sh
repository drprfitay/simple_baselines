#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"
use_venv=1
skip_deps=0
check_only=0
model_check=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --python PATH       Python interpreter used to create/run the environment
  --venv PATH         Virtual-environment directory (default: .venv)
  --no-venv           Install into the selected active environment
  --skip-deps         Do not install requirements.txt
  --check-only        Run verification without installing anything
  --model-check       Also load the bundled ESM2 8M checkpoint
  -h, --help          Show this help

Environment alternatives: PYTHON=/path/to/python and VENV_DIR=/path/to/venv.
EOF
}

while (($#)); do
  case "$1" in
    --python)
      python_bin="$2"
      shift 2
      ;;
    --venv)
      venv_dir="$2"
      shift 2
      ;;
    --no-venv)
      use_venv=0
      shift
      ;;
    --skip-deps)
      skip_deps=1
      shift
      ;;
    --check-only)
      check_only=1
      use_venv=0
      skip_deps=1
      shift
      ;;
    --model-check)
      model_check=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$python_bin" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter not found: $python_bin" >&2
  exit 1
fi

"$python_bin" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise SystemExit("Python 3.10 or 3.11 is required; found " + sys.version.split()[0])
PY

if ((check_only == 0 && use_venv == 1)); then
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "$python_bin" -m venv "$venv_dir"
  fi
  python_bin="${venv_dir}/bin/python"
fi

if ((check_only == 0)); then
  if ((skip_deps == 0)); then
    "$python_bin" -m pip install --upgrade pip setuptools wheel
    "$python_bin" -m pip install -r "${repo_root}/requirements.txt"
  fi
  "$python_bin" -m pip install --no-deps --no-build-isolation -e "$repo_root"
fi

verify_args=(--skip-data)
if ((model_check == 1)); then
  verify_args+=(--model-check)
fi
MPLCONFIGDIR="${repo_root}/.matplotlib" "$python_bin" "${repo_root}/scripts/verify_install.py" "${verify_args[@]}"

echo
echo "Code installation verified."
if ((use_venv == 1)); then
  echo "Activate it with: source ${venv_dir}/bin/activate"
fi
echo "After downloading the data, verify it with: python scripts/verify_install.py --strict-data"
echo "Start notebooks from the repository root with: jupyter lab"
