VERSION=v4
LOGFILE=logs/exp_${VER5ION}.log

CUDA_VISIBLE_DEVICES="0" python3 train.py > "$LOGFILE" 2>&1 &