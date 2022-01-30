VERSION=v2
LOGFILE=logs/exp_${VERSION}.log

CUDA_VISIBLE_DEVICES="0" python decision_tree.py > "$LOGFILE" 2>&1 &