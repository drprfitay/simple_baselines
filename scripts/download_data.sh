#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The release is split into two archives. Keep the old environment variable
# names as fallbacks so existing setup commands continue to work.
data_url="${DATA_URL:-${DATASETS_FOR_ANALYSIS_URL:-https://drive.google.com/file/d/1CqldNuKitdeafFbo65sfGOn3vSFIy_s5/view?usp=sharing}}"
merged_embeddings_url="${MERGED_DATASETS_AND_EMBEDDINGS_URL:-${MERGED_DATASETS_URL:-https://drive.google.com/file/d/1P0JFKBk5s3BXT5Jo983l_rlFiAPrpfLG/view?usp=sharing}}"
download_dir="${DOWNLOAD_DIR:-}"

usage() {
  cat <<'EOF'
Usage: scripts/download_data.sh [options]

Downloads and installs the two prepared-data archives:
  1. data_final.zip: datasets are installed under data/
  2. merged_datasets_and_embeddings.zip: merged_datasets/ is installed at the
     repository root, and each dataset's embeddings are installed under
     data/<dataset>/embeddings/

Options:
  --data-url URL               Override the data archive URL
  --merged-embeddings-url URL  Override the merged-datasets/embeddings URL
  --download-dir DIR           Keep downloaded archives in DIR instead of a
                               temporary directory
  -h, --help                   Show this help

URLs can also be overridden with DATA_URL and
MERGED_DATASETS_AND_EMBEDDINGS_URL. DOWNLOAD_DIR is useful for retaining the
large archives for reuse.
EOF
}

while (($#)); do
  case "$1" in
    --data-url)
      [[ $# -ge 2 ]] || { echo "--data-url requires a value" >&2; exit 2; }
      data_url="$2"
      shift 2
      ;;
    --merged-embeddings-url)
      [[ $# -ge 2 ]] || { echo "--merged-embeddings-url requires a value" >&2; exit 2; }
      merged_embeddings_url="$2"
      shift 2
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

# Stage extraction on the repository filesystem. Fresh installations can then
# move large directories into place instead of making a second multi-GB copy.
extract_dir="$(mktemp -d "$repo_root/.simple-baselines-extract.XXXXXX")"
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
  extract_dir="$(mktemp -d "$repo_root/.simple-baselines-extract.XXXXXX")"
}

install_directory() {
  local source_dir="$1"
  local destination_dir="$2"

  mkdir -p "$(dirname "$destination_dir")"
  if [[ ! -e "$destination_dir" && ! -L "$destination_dir" ]]; then
    mv -- "$source_dir" "$destination_dir"
  elif [[ -d "$destination_dir" ]]; then
    cp -a "$source_dir/." "$destination_dir/"
  else
    echo "Installation destination is not a directory: $destination_dir" >&2
    exit 1
  fi
}

install_data() {
  local archive="$1"
  local source_dir=""

  reset_extract_dir
  unzip -q "$archive" -d "$extract_dir"

  # data_final/ is the release layout; data/ is accepted for compatibility
  # with locally repacked archives.
  for candidate in "$extract_dir/data_final" "$extract_dir/data"; do
    if [[ -d "$candidate" ]]; then
      source_dir="$candidate"
      break
    fi
  done

  if [[ -z "$source_dir" ]]; then
    echo "Expected a data_final/ directory in $archive" >&2
    exit 1
  fi

  install_directory "$source_dir" "$repo_root/data"
  echo "Installed datasets in $repo_root/data"
}

install_merged_datasets_and_embeddings() {
  local archive="$1"
  local merged_source
  local embeddings_source
  local dataset_source
  local dataset_name
  local installed_embeddings=0

  reset_extract_dir
  unzip -q "$archive" -d "$extract_dir"
  merged_source="$extract_dir/merged_datasets"
  embeddings_source="$extract_dir/embeddings"

  if [[ ! -d "$merged_source" ]]; then
    echo "Expected a merged_datasets/ directory in $archive" >&2
    exit 1
  fi
  if [[ ! -d "$embeddings_source" ]]; then
    echo "Expected an embeddings/ directory in $archive" >&2
    exit 1
  fi

  install_directory "$merged_source" "$repo_root/merged_datasets"
  echo "Installed merged datasets in $repo_root/merged_datasets"

  while IFS= read -r -d '' dataset_source; do
    dataset_name="$(basename "$dataset_source")"
    if [[ ! -d "$dataset_source/embeddings" ]]; then
      echo "Expected embeddings/$dataset_name/embeddings/ in $archive" >&2
      exit 1
    fi

    install_directory \
      "$dataset_source/embeddings" \
      "$repo_root/data/$dataset_name/embeddings"
    echo "Installed $dataset_name embeddings in $repo_root/data/$dataset_name/embeddings"
    installed_embeddings=$((installed_embeddings + 1))
  done < <(find "$embeddings_source" -mindepth 1 -maxdepth 1 -type d -print0)

  if ((installed_embeddings == 0)); then
    echo "No dataset embedding directories found in $archive" >&2
    exit 1
  fi

  # Some analysis code still refers to data/merged_datasets. A relative link
  # keeps that path working without storing a second copy of the large tables.
  local legacy_merged_path="$repo_root/data/merged_datasets"
  if [[ ! -e "$legacy_merged_path" && ! -L "$legacy_merged_path" ]]; then
    ln -s ../merged_datasets "$legacy_merged_path"
    echo "Linked compatibility path: $legacy_merged_path"
  elif [[ -L "$legacy_merged_path" && "$(readlink "$legacy_merged_path")" == "../merged_datasets" ]]; then
    :
  else
    echo "Compatibility path already exists; left unchanged: $legacy_merged_path"
  fi
}

data_archive="$(download_archive "$data_url" data_final.zip)"
install_data "$data_archive"

merged_embeddings_archive="$(download_archive "$merged_embeddings_url" merged_datasets_and_embeddings.zip)"
install_merged_datasets_and_embeddings "$merged_embeddings_archive"

echo "Data setup complete."
