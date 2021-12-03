VERSION=v1
LOGFILE=logs/exp_${VERSION}.log

python3 lin_reg.py > "$LOGFILE" 2>&1 &