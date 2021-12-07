VERSION=v4
LOGFILE=logs/exp_${VERSION}.log

#python3 log_reg.py > "$LOGFILE" 2>&1 &
CUDA_VISIBLE_DEVICES="0, 1, 2, 3" python3 log_reg_mp.py > "$LOGFILE" 2>&1 &
#CUDA_VISIBLE_DEVICES="-1" python3 log_reg.py > "$LOGFILE" 2>&1 &