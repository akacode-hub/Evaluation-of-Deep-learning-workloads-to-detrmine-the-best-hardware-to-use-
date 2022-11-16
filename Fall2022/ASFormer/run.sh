dataset=breakfast #50salads
split=1
action=train #predict
version=exp4
LOGFILE=loggers/${dataset}_${split}_${action}_${version}.log

# train
CUDA_VISIBLE_DEVICES=2 python3 main.py --action=${action} --dataset=${dataset} --split=${split} > "$LOGFILE" 2>&1 & 


# predict
# CUDA_VISIBLE_DEVICES=2 python3 main.py --action=${action} --dataset=${dataset} --split=${split} > "$LOGFILE" 2>&1 &

# eval
# CUDA_VISIBLE_DEVICES=2 python3 eval.py --dataset=${dataset} --split=${split}  > "$LOGFILE" 2>&1 &
