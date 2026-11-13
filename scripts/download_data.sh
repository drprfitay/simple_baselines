#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

merged_datasets_url="${MERGED_DATASETS_URL:-https://drive.google.com/file/d/18KK9ig260Hzwl1rVgqRnjOyiCbopD2qw/view?usp=drive_link}"
datasets_for_analysis_url="${DATASETS_FOR_ANALYSIS_URL:-https://drive.google.com/file/d/16s7qcJCjR4TsbVn-ShWe3ycdxXo84aEC/view?usp=drive_link}"
embeddings_url="${EMBEDDINGS_URL:-}"
download_embeddings=0
download_dir="${DOWNLOAD_DIR:-}"

usage() {
  cat <<'EOF'
Usage: scripts/download_data.sh [options]

Downloads and installs the analysis datasets and merged datasets. The
embedding archive is optional until its Google Drive link is available.

Options:
  --embeddings-url URL  Also download and install embeddings.zip
  --embeddings-only     Skip the two dataset archives (requires an embedding URL)
  --download-dir DIR    Keep downloaded archives in DIR instead of a temporary directory
  -h, --help            Show this help

The embedding URL can also be supplied through EMBEDDINGS_URL. DOWNLOAD_DIR
is useful for retaining large archives or resuming an interrupted setup.
EOF
}

download_datasets=1
while (($#)); do
  case "$1" in
    --embeddings-url)
      [[ $# -ge 2 ]] || { echo "--embeddings-url requires a value" >&2; exit 2; }
      embeddings_url="$2"
      download_embeddings=1
      shift 2
      ;;
    --embeddings-only)
      download_datasets=0
      download_embeddings=1
      shift
      ;;
    --download-dir)
      [[ $# -ge 2 ]] || { echo "--download-dir requires a value" >&2; exit 2; }
      download_dir="$2"
      shift 2
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

if [[ -n "$embeddings_url" ]]; then
  download_embeddings=1
fi
if ((download_embeddings == 1)) && [[ -z "$embeddings_url" ]]; then
  echo "An embedding link is required; pass --embeddings-url or set EMBEDDINGS_URL." >&2
  exit 2
fi

for command_name in curl unzip; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

temporary_download_dir=0
if [[ -z "$download_dir" ]]; then
  download_dir="$(mktemp -d "${TMPDIR:-/tmp}/simple-baselines-download.XXXXXX")"
  temporary_download_dir=1
else
  mkdir -p "$download_dir"
  download_dir="$(cd "$download_dir" && pwd)"
fi

extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/simple-baselines-extract.XXXXXX")"
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
    BEGIN { unsafe = 0 }
    /^\// || /(^|\/)\.\.($|\/)/ { unsafe = 1 }
    END { exit unsafe ? 0 : 1 }
  '; then
    echo "Refusing to extract unsafe archive paths from: $archive" >&2
    exit 1
  fi
}

download_archive() {
  local source_url="$1"
  local archive_name="$2"
  local archive_path="${download_dir}/${archive_name}"
  local resolved_url
  resolved_url="$(google_drive_download_url "$source_url")"

  if [[ -f "$archive_path" ]] && unzip -tq "$archive_path" >/dev/null 2>&1; then
    echo "Using existing archive: $archive_path" >&2
  else
    echo "Downloading $archive_name ..." >&2
    rm -f -- "${archive_path}.part"
    curl --fail --location --retry 3 --retry-all-errors \
      --output "${archive_path}.part" "$resolved_url"
    mv -- "${archive_path}.part" "$archive_path"
  fi

  validate_zip "$archive_path"
  printf '%s\n' "$archive_path"
}

reset_extract_dir() {
  rm -rf -- "$extract_dir"
  extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/simple-baselines-extract.XXXXXX")"
}

install_datasets_for_analysis() {
  local archive="$1"
  reset_extract_dir
  unzip -q "$archive" -d "$extract_dir"
  if [[ ! -d "$extract_dir/data" ]]; then
    echo "Expected a data/ directory in $archive" >&2
    exit 1
  fi
  mkdir -p "$repo_root/data"
  cp -a "$extract_dir/data/." "$repo_root/data/"
  echo "Installed datasets for analysis in $repo_root/data"
}

install_merged_datasets() {
  local archive="$1"
  local source_dir=""
  reset_extract_dir
  unzip -q "$archive" -d "$extract_dir"

  for candidate in \
    "$extract_dir/data/merged_datasets" \
    "$extract_dir/notebooks/data/merged_datasets" \
    "$extract_dir/merged_datasets"; do
    if [[ -d "$candidate" ]]; then
      source_dir="$candidate"
      break
    fi
  done

  if [[ -z "$source_dir" ]]; then
    echo "Could not find merged_datasets/ in $archive" >&2
    exit 1
  fi

  mkdir -p "$repo_root/data/merged_datasets"
  cp -a "$source_dir/." "$repo_root/data/merged_datasets/"
  echo "Installed merged datasets in $repo_root/data/merged_datasets"

  local notebook_data_path="$repo_root/merged_datasets"
  if [[ -e "$notebook_data_path" || -L "$notebook_data_path" ]]; then
    echo "Notebook data path already exists; left unchanged: $notebook_data_path"
  else
    ln -s data/merged_datasets "$notebook_data_path"
    echo "Linked notebook data path: $notebook_data_path"
  fi
}

install_embeddings() {
  local archive="$1"
  local required_path
  local required_paths=(
    data/pard3/embeddings/esm_35m
    data/pard3/embeddings/esm_150m
    data/pard3/embeddings/esm_650m
    data/casp/embeddings/saprot
    data/casp/embeddings/saprot_aav_sites
    data/his2/embeddings/prot_bert
    data/his2/embeddings/progen2-small
  )

  reset_extract_dir
  unzip -q "$archive" -d "$extract_dir"
  for required_path in "${required_paths[@]}"; do
    if [[ ! -d "$extract_dir/$required_path" ]]; then
      echo "Expected $required_path in $archive" >&2
      exit 1
    fi
  done

  mkdir -p "$repo_root/data"
  cp -a "$extract_dir/data/." "$repo_root/data/"
  echo "Installed PARD3, CASP, and HIS2 embeddings in $repo_root/data"
}

if ((download_datasets == 1)); then
  datasets_archive="$(download_archive "$datasets_for_analysis_url" datasets_for_analysis.zip)"
  install_datasets_for_analysis "$datasets_archive"

  merged_archive="$(download_archive "$merged_datasets_url" merged_datasets.zip)"
  install_merged_datasets "$merged_archive"
fi

if ((download_embeddings == 1)); then
  embeddings_archive="$(download_archive "$embeddings_url" embeddings.zip)"
  install_embeddings "$embeddings_archive"
else
  echo "Embedding archive skipped; pass --embeddings-url after uploading embeddings.zip."
fi

echo "Data setup complete."
