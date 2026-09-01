#!/usr/bin/env bash
set -euo pipefail

release_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
mode="${1:-full}"
if [[ "$mode" == "full" || "$mode" == "quick" || "$mode" == "dry-run" ]]; then
  shift || true
else
  echo "usage: $0 {full|quick|dry-run} (--expected-git-commit COMMIT | --expected-release-inventory-sha256 SHA256) [--existing-python PATH] [runner options]" >&2
  exit 2
fi

existing_python=""
expected_inventory=""
expected_commit=""
requested_receipt=""
forwarded=()
while (($#)); do
  case "$1" in
    --existing-python)
      [[ $# -ge 2 ]] || { echo "--existing-python requires a path" >&2; exit 2; }
      existing_python="$2"
      shift 2
      ;;
    --expected-release-inventory-sha256)
      [[ $# -ge 2 ]] || { echo "--expected-release-inventory-sha256 requires a digest" >&2; exit 2; }
      expected_inventory="$2"
      shift 2
      ;;
    --expected-git-commit)
      [[ $# -ge 2 ]] || { echo "--expected-git-commit requires an object ID" >&2; exit 2; }
      expected_commit="$2"
      shift 2
      ;;
    --receipt)
      [[ $# -ge 2 ]] || { echo "--receipt requires a path" >&2; exit 2; }
      requested_receipt="$2"
      shift 2
      ;;
    --assembly-workers|--surface-workers)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      forwarded+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "unsupported or protected wrapper option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$expected_inventory" && -n "$expected_commit" ]] || \
   [[ -z "$expected_inventory" && -z "$expected_commit" ]]; then
  echo "supply exactly one externally trusted --expected-release-inventory-sha256 or --expected-git-commit" >&2
  exit 2
fi
identity_arguments=()
if [[ -n "$expected_inventory" ]]; then
  identity_arguments+=(--expected-inventory-sha256 "$expected_inventory")
  runner_identity_arguments=(--expected-release-inventory-sha256 "$expected_inventory")
else
  identity_arguments+=(--expected-git-commit "$expected_commit")
  runner_identity_arguments=(--expected-git-commit "$expected_commit")
fi

bootstrap_python="$(command -v python3 || true)"
if [[ -z "$bootstrap_python" ]]; then
  echo "python3 is required for the stdlib-only authenticated release preflight" >&2
  exit 1
fi
"$bootstrap_python" -I -B \
  "$release_root/Numerics/stage_blaschke_certificate_replay.py" \
  --source-root "$release_root" --check-only "${identity_arguments[@]}"

export PYTHONDONTWRITEBYTECODE=1
kernel_name="blaschke-certificate-replay"

if [[ -n "$existing_python" ]]; then
  replay_python="$existing_python"
else
  replay_prefix="$release_root/.certificate-replay-env"
  replay_python="$replay_prefix/bin/python"
  if [[ ! -x "$replay_python" ]]; then
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
      environment_tool="${CONDA_EXE}"
    elif command -v conda >/dev/null 2>&1; then
      environment_tool="$(command -v conda)"
    elif command -v micromamba >/dev/null 2>&1; then
      environment_tool="$(command -v micromamba)"
    else
      echo "conda or micromamba is required to create the exact locked environment" >&2
      exit 1
    fi
    "$environment_tool" create --yes --prefix "$replay_prefix" \
      --file "$release_root/conda-explicit-lock.txt"
    "$replay_python" -m ensurepip --upgrade
    "$replay_python" -m pip install --no-deps --only-binary=:all: \
      --require-hashes --requirement "$release_root/pip-requirements-lock.txt"
  fi
fi

if [[ "$mode" == "quick" ]]; then
  quick_arguments=(--root "$release_root" --mode artifact-only "${runner_identity_arguments[@]}")
  if [[ -n "$requested_receipt" ]]; then
    quick_arguments+=(--receipt "$requested_receipt")
  fi
  exec "$replay_python" -I -B \
    "$release_root/Numerics/verify_blaschke_certificate_equivalence.py" \
    "${quick_arguments[@]}"
fi

if [[ "$mode" == "full" ]]; then
  memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  disk_kib="$(df -Pk "${TMPDIR:-/tmp}" | awk 'NR==2 {print $4}')"
  if ((memory_kib < 8 * 1024 * 1024)); then
    echo "full replay requires at least 8 GiB RAM" >&2
    exit 1
  fi
  if ((disk_kib < 8 * 1024 * 1024)); then
    echo "full replay requires at least 8 GiB free scratch disk" >&2
    exit 1
  fi
fi

replay_workspace="$(mktemp -d "${TMPDIR:-/tmp}/blaschke-certificate-replay.XXXXXXXX")"
receipt_path="${requested_receipt:-$replay_workspace/certificate-equivalence-receipt.json}"
runner_arguments=(
  --source-root "$release_root"
  --workspace "$replay_workspace"
  --existing-python "$replay_python"
  --kernel-name "$kernel_name"
  --receipt "$receipt_path"
  "${runner_identity_arguments[@]}"
)
if [[ "$mode" == "dry-run" ]]; then
  runner_arguments+=(--dry-run)
fi
"$replay_python" -I -B \
  "$release_root/Numerics/run_blaschke_certificate_reproduction.py" \
  "${runner_arguments[@]}" "${forwarded[@]}"
echo "receipt: $receipt_path" >&2
