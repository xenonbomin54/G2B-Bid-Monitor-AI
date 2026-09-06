#!/bin/zsh
cd /Users/xenonbomin54/나라장터
export PPS_API_BASE="https://integrate.api.nvidia.com/v1" PPS_API_MODEL="google/gemma-4-31b-it" PPS_API_RPM=40
export PPS_API_KEY=$(grep -o 'nvapi-[A-Za-z0-9_-]*' apiapiapi.py | head -1)
while pgrep -f "dev200c_pred" >/dev/null; do sleep 30; done
# 판로지원 예외 지침 + 2단계 검증
python3 -u tools/evaluate.py --runner api --verify --save out/dev200d_pred.csv > out/dev200d_eval.log 2>&1
# 실제 분포 홀드아웃 (라벨 없음 → 예측률만)
python3 -u tools/evaluate.py --runner api --verify --data out/holdout200.jsonl.gz \
  --labels /dev/null --save out/holdout_pred.csv > out/holdout_eval.log 2>&1
