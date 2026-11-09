#!/usr/bin/env bash
# Clone the official baseline implementations into external/, pinned to one commit each.
# experiments/train_bundle.py imports model code from here; it owns data, seeds and evaluation.
#
# Usage: bash scripts/setup_baselines.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="${ROOT}/external"
mkdir -p "${EXT}"

REPOS=(
  "CrossCBR https://github.com/mysbupt/CrossCBR bbe80dfb1bbe878d72389185edefc655c8b5db45"
  "MultiCBR https://github.com/HappyPointer/MultiCBR 744cec8e22f193631d8bdd84050f00edf6f11184"
  "bundle-recommendation-survey https://github.com/WUT-IDEA/bundle-recommendation-survey e43c9a281398c81b5c9284ab7df17461a29f1efe"
  "CoHeat https://github.com/snudatalab/CoHeat f1879d502302228c5ad4f3b09a9024e8f549f7ce"
)

for entry in "${REPOS[@]}"; do
  read -r name url commit <<<"${entry}"
  dest="${EXT}/${name}"
  if [[ ! -d "${dest}/.git" ]]; then
    # Code only: skip large blobs and the bundled dataset archives.
    git clone -q --filter=blob:limit=2m --no-checkout "${url}" "${dest}"
    git -C "${dest}" sparse-checkout set --no-cone '/*' '!*.tgz' '!*.zip' '!*.tar.gz' '!*.pkl'
  fi
  git -C "${dest}" checkout -q "${commit}"
  echo "${name} @ ${commit:0:12}"
done
