#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " Validating Yi-Lightning MoE Reference Runtime    "
echo "=================================================="

python3 -m py_compile moe_expert_router.py
python3 -m py_compile bilingual_token_estimator.py

python3 moe_expert_router.py
python3 bilingual_token_estimator.py

echo "Validation successful!"
