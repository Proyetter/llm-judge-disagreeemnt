#!/usr/bin/env bash
# Regenerates everything in results/ from the saved judgments in data/.
# No API key needed. No paid calls are made.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/data"

run () {
  echo "=== $1 -> results/$2 ==="
  python "$ROOT/scripts/$1" > "$ROOT/results/$2" 2>&1
}

run audit_factorial.py           audit.txt
run analyze_decomposition.py     table2_decomposition.txt
run analyze_pair_classes.py      table3_instability_categories.txt
run analyze_conditional_noise.py table3_conditional_noise.txt
run analyze_confidence_v2.py     table4_confidence_auroc.txt

echo
echo "Done. Outputs are in results/."
