#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLUGIN_ROOT="${ROOT}/plugin/work-receipts"
PLUGIN_SKILL="${PLUGIN_ROOT}/skills/work-receipts"
DIST="${ROOT}/dist"
STAGING="$(mktemp -d "${TMPDIR:-/tmp}/work-receipts-package.XXXXXX")"

cleanup() {
  rm -rf "${STAGING}"
}
trap cleanup EXIT

if [[ ! -f "${ROOT}/SKILL.md" || ! -f "${PLUGIN_ROOT}/.codex-plugin/plugin.json" ]]; then
  echo "Missing Work Receipts skill or plugin manifest." >&2
  exit 1
fi

# The canonical skill lives at WorkReceipts/. Refresh the plugin's embedded copy.
rm -rf "${PLUGIN_SKILL}"
mkdir -p "${PLUGIN_SKILL}"
cp "${ROOT}/SKILL.md" "${PLUGIN_SKILL}/SKILL.md"
cp -R "${ROOT}/references" "${PLUGIN_SKILL}/references"
cp -R "${ROOT}/assets" "${PLUGIN_SKILL}/assets"

mkdir -p "${DIST}"
rm -f "${DIST}/work-receipts.zip" "${DIST}/work-receipts-chatgpt-plugin.zip"

mkdir -p "${STAGING}/skill/work-receipts"
cp "${ROOT}/SKILL.md" "${STAGING}/skill/work-receipts/SKILL.md"
cp -R "${ROOT}/references" "${STAGING}/skill/work-receipts/references"
cp -R "${ROOT}/assets" "${STAGING}/skill/work-receipts/assets"

(
  cd "${STAGING}/skill"
  zip -X -q -r "${DIST}/work-receipts.zip" work-receipts
)

mkdir -p "${STAGING}/plugin"
cp -R "${PLUGIN_ROOT}" "${STAGING}/plugin/work-receipts"

(
  cd "${STAGING}/plugin"
  zip -X -q -r "${DIST}/work-receipts-chatgpt-plugin.zip" work-receipts
)

echo "Created:"
echo "  ${DIST}/work-receipts.zip"
echo "  ${DIST}/work-receipts-chatgpt-plugin.zip"
