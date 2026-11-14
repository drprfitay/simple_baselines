#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
progen_url="${PROGEN_WEIGHTS_URL:-https://drive.google.com/file/d/1yFH3tYVabuY0wwhPhcBaKWiyFLBakeCb/view?usp=sharing}"
download_dir="${DOWNLOAD_DIR:-}"
destination_dir="$repo_root/models/weights/progen"
force=0

expected_files=(
  progen2-small_config.json
  progen2-small_tokenizer.json
  progen2-small_weights.pth
  progen2-medium_config.json
  progen2-medium_tokenizer.json
  progen2-medium_weights.pth
)

usage() {
  cat <<'EOF'
Usage: scripts/download_progen_weights.sh [options]

Downloads the optional ProGen2 checkpoint archive and installs the small and
medium model files under models/weights/progen/. This script is intentionally
separate from install.sh and scripts/download_data.sh.

Options:
  --url URL             Override the ProGen weights archive URL
  --download-dir DIR    Keep progen.zip in DIR instead of deleting it after
                        installation
  --force               Reinstall even when all expected files already exist
  -h, --help            Show this help

The URL and retained-download directory can also be set with
PROGEN_WEIGHTS_URL and DOWNLOAD_DIR, respectively.
EOF
}

while (($#)); do
  case "$1" in
    --url)
      [[ $# -ge 2 ]] || { echo "--url requires a value" >&2; exit 2; }
      progen_url="$2"
      shift 2
      ;;
    --download-dir)
      [[ $# -ge 2 ]] || { echo "--download-dir requires a value" >&2; exit 2; }
      download_dir="$2"
      shift 2
      ;;
    --force)
      force=1
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

for command_name in curl unzip; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

if ((force == 0)); then
  installation_complete=1
  for filename in "${expected_files[@]}"; do
    if [[ ! -f "$destination_dir/$filename" ]]; then
      installation_complete=0
      break
    fi
  done
  if ((installation_complete == 1)); then
    echo "ProGen weights are already installed in $destination_dir"
    echo "Use --force to download and reinstall them."
    exit 0
  fi
fi

temporary_download_dir=0
if [[ -z "$download_dir" ]]; then
  download_dir="$(mktemp -d "$repo_root/.progen-download.XXXXXX")"
  temporary_download_dir=1
else
  mkdir -p "$download_dir"
  download_dir="$(cd "$download_dir" && pwd)"
fi

extract_dir="$(mktemp -d "$repo_root/.progen-extract.XXXXXX")"
cleanup() {
  rm -rf -- "$extract_dir"
  if ((temporary_download_dir == 1)); then
    rm -rf -- "$download_dir"
  fi
}
trap cleanup EXIT

google_drive_download_url() {
  local source_url="$1"
  local file_id=""

  if [[ "$source_url" =~ /d/([^/?]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  elif [[ "$source_url" =~ [\?\&]id=([^\&]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  fi

  if [[ -n "$file_id" ]]; then
    printf 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t\n' "$file_id"
  else
    printf '%s\n' "$source_url"
  fi
}

validate_zip() {
  local archive="$1"
  if ! unzip -tq "$archive" >/dev/null; then
    echo "Downloaded file is not a valid ZIP archive: $archive" >&2
    echo "Check that the Google Drive file is shared with anyone who has the link." >&2
    exit 1
  fi

  if unzip -Z1 "$archive" | awk '
    /^\// || /(^|\/)\.\.($|\/)/ { unsafe = 1 }
    END { exit unsafe ? 0 : 1 }
  '; then
    echo "Refusing to extract unsafe archive paths from: $archive" >&2
    exit 1
  fi
}

archive_path="$download_dir/progen.zip"
if [[ -f "$archive_path" ]] && unzip -tq "$archive_path" >/dev/null 2>&1; then
  echo "Using existing archive: $archive_path"
else
  resolved_url="$(google_drive_download_url "$progen_url")"
  echo "Downloading ProGen weights (about 3.2 GiB) ..."
  rm -f -- "$archive_path.part"
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$archive_path.part" "$resolved_url"
  mv -- "$archive_path.part" "$archive_path"
fi

validate_zip "$archive_path"
echo "Extracting ProGen weights ..."
unzip -q "$archive_path" -d "$extract_dir"

progen_source=""
while IFS= read -r candidate_dir; do
  candidate_complete=1
  for filename in "${expected_files[@]}"; do
    if [[ ! -f "$candidate_dir/$filename" ]]; then
      candidate_complete=0
      break
    fi
  done
  if ((candidate_complete == 1)); then
    progen_source="$candidate_dir"
    break
  fi
done < <(find "$extract_dir" -type f -name progen2-small_weights.pth -printf '%h\n')

if [[ -z "$progen_source" ]]; then
  echo "The archive does not contain the expected ProGen2 small and medium model files." >&2
  exit 1
fi

mkdir -p "$destination_dir"
for filename in "${expected_files[@]}"; do
  mv -- "$progen_source/$filename" "$destination_dir/$filename"
done

echo "Installed ProGen weights in $destination_dir"
