#!/bin/zsh
cd /Users/xenonbomin54/나라장터
export PPS_API_BASE="https://integrate.api.nvidia.com/v1" PPS_API_MODEL="google/gemma-4-31b-it" PPS_API_RPM=40
export PPS_API_KEY=$(grep -o 'nvapi-[A-Za-z0-9_-]*' apiapiapi.py | head -1)
while pgrep -f "dev200f_pred" >/dev/null; do sleep 20; done
python3 -u tools/evaluate.py --runner api --save out/dev200g_pred.csv > out/dev200g_eval.log 2>&1
