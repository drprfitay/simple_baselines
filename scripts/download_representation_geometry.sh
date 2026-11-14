#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
geometry_url="${REPRESENTATION_GEOMETRY_URL:-https://drive.google.com/file/d/1oNZ9k0BS1wzgNX0K38S0tmFhK7Q4gxqd/view?usp=sharing}"
download_dir="${DOWNLOAD_DIR:-}"
force=0

expected_files=(
  data/casp/external_scoring/new_features.pt
  data/gfp/external_scoring/new_features.pt
  data/his2/external_scoring/new_features.pt
  data/his5/external_scoring/new_features.pt
  data/lov/external_scoring/new_features.pt
  data/nmt/external_scoring/new_features.pt
  data/pard3/external_scoring/new_features.pt
)

usage() {
  cat <<'EOF'
Usage: scripts/download_representation_geometry.sh [options]

Downloads the prepared representation-geometry feature archive and installs
its tensors under data/<dataset>/external_scoring/.

Options:
  --url URL             Override the representation-geometry archive URL
  --download-dir DIR    Keep the downloaded archive in DIR instead of deleting
                        it after installation
  --force               Reinstall even when all expected tensors already exist
  -h, --help            Show this help

The URL and retained-download directory can also be set with
REPRESENTATION_GEOMETRY_URL and DOWNLOAD_DIR, respectively.
EOF
}

while (($#)); do
  case "$1" in
    --url)
      [[ $# -ge 2 ]] || { echo "--url requires a value" >&2; exit 2; }
      geometry_url="$2"
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
  for relative_path in "${expected_files[@]}"; do
    if [[ ! -f "$repo_root/$relative_path" ]]; then
      installation_complete=0
      break
    fi
  done
  if ((installation_complete == 1)); then
    echo "Representation-geometry features are already installed under $repo_root/data"
    echo "Use --force to download and reinstall them."
    exit 0
  fi
fi

temporary_download_dir=0
if [[ -z "$download_dir" ]]; then
  download_dir="$(mktemp -d "$repo_root/.representation-geometry-download.XXXXXX")"
  temporary_download_dir=1
else
  mkdir -p "$download_dir"
  download_dir="$(cd "$download_dir" && pwd)"
fi

extract_dir="$(mktemp -d "$repo_root/.representation-geometry-extract.XXXXXX")"
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
  local contents_validated="${2:-0}"
  if ((contents_validated == 0)) && ! unzip -tq "$archive" >/dev/null; then
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

archive_path="$download_dir/representation_geometry_new_features.zip"
archive_contents_validated=0
if [[ -f "$archive_path" ]] && unzip -tq "$archive_path" >/dev/null 2>&1; then
  archive_contents_validated=1
  echo "Using existing archive: $archive_path"
else
  resolved_url="$(google_drive_download_url "$geometry_url")"
  echo "Downloading representation-geometry features (about 2.7 GiB) ..."
  rm -f -- "$archive_path.part"
  curl --fail --location --retry 3 --retry-all-errors \
    --output "$archive_path.part" "$resolved_url"
  mv -- "$archive_path.part" "$archive_path"
fi

validate_zip "$archive_path" "$archive_contents_validated"
echo "Extracting representation-geometry features ..."
unzip -q "$archive_path" -d "$extract_dir"

# Accept either the published data/... layout or the same layout nested under
# a single wrapper directory.
casp_source="$(find "$extract_dir" -type f \
  -path '*/data/casp/external_scoring/new_features.pt' -print -quit)"
if [[ -z "$casp_source" ]]; then
  echo "The archive does not contain the expected representation-geometry tensors." >&2
  exit 1
fi
source_root="${casp_source%/data/casp/external_scoring/new_features.pt}"

for relative_path in "${expected_files[@]}"; do
  if [[ ! -f "$source_root/$relative_path" ]]; then
    echo "Missing expected archive member: $relative_path" >&2
    exit 1
  fi
done

for relative_path in "${expected_files[@]}"; do
  destination="$repo_root/$relative_path"
  mkdir -p "$(dirname "$destination")"
  mv -f -- "$source_root/$relative_path" "$destination"
done

echo "Installed representation-geometry features under $repo_root/data"
