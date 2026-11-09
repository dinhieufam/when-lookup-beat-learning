#!/usr/bin/env bash
# Download MealRec+ (H and L) from WUT-IDEA/MealRecPlus, pinned to one commit.
#
# Usage:
#   bash scripts/download_mealrecplus.sh          # relation + healthiness files (~5 MB)
#   bash scripts/download_mealrecplus.sh --meta   # also the Git LFS metadata zips (several hundred MB)
#
# Files land in data/raw/MealRec+H and data/raw/MealRec+L, mirroring the upstream layout.
# Hashes of the result are listed in data/README.md; check with --verify.
set -euo pipefail

COMMIT=8bd7d7d7effb505be4a7d2074b28abad802fe0f7
RAW="https://raw.githubusercontent.com/WUT-IDEA/MealRecPlus/${COMMIT}"
LFS="https://media.githubusercontent.com/media/WUT-IDEA/MealRecPlus/${COMMIT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data/raw"

META=0
VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --meta) META=1 ;;
    --verify) VERIFY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

FILES=(
  course_category.txt
  meal_course.txt
  user_course.txt
  user_meal_train.txt
  user_meal_tune.txt
  user_meal_test.txt
  healthiness/course_fsa.txt
  healthiness/course_who.txt
  healthiness/meal_fsa.txt
  healthiness/meal_who.txt
  healthiness/user_fsa.txt
  healthiness/user_who.txt
  meta_data/course2index.txt
  meta_data/user2index.txt
)
META_FILES=(meta_data/course.zip meta_data/user_course.zip)

fetch() {  # url dest
  mkdir -p "$(dirname "$2")"
  if [[ -s "$2" ]]; then return; fi
  echo "  ${2#"$ROOT"/}"
  curl -fsSL --retry 3 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

for ds in "MealRec+H" "MealRec+L"; do
  enc="${ds//+/%2B}"
  echo "${ds}"
  for f in "${FILES[@]}"; do
    fetch "${RAW}/MealRec%2B/${enc}/${f}" "${OUT}/${ds}/${f}"
  done
  if [[ $META -eq 1 ]]; then
    for f in "${META_FILES[@]}"; do
      fetch "${LFS}/MealRec%2B/${enc}/${f}" "${OUT}/${ds}/${f}"
    done
  fi
done

# Upstream reference code, kept for cross-checking our loader and metrics.
fetch "${RAW}/MealRec%2B/data_load.py" "${OUT}/reference/data_load.py"
fetch "${RAW}/healthiness_eval.py" "${OUT}/reference/healthiness_eval.py"

if [[ $VERIFY -eq 1 ]]; then
  (cd "${OUT}" && shasum -a 256 -c "${ROOT}/data/SHA256SUMS" --ignore-missing)
fi
echo "done -> ${OUT}"
