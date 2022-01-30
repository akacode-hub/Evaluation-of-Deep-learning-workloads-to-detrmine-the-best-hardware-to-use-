VERSION=v1
LOGFILE=logs/exp_${VERSION}.log

CUDA_VISIBLE_DEVICES="0" python3 train.py > "$LOGFILE" 2>&1 &
#CUDA_VISIBLE_DEVICES="0, 3" python3 train_amp.py > "$LOGFILE" 2>&1 &