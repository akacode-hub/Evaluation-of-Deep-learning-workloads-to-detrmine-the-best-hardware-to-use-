dataset=breakfast #50salads
split=1
action=train #predict
version=exp5
arch=ddl
LOGFILE=loggers/${dataset}_${split}_${action}_${arch}_${version}.log

# train
CUDA_VISIBLE_DEVICES=0 python3 main.py --action=${action} --dataset=${dataset} --version=${version} --arch=${arch} --split=${split} > "$LOGFILE" 2>&1 & 


# predict
# CUDA_VISIBLE_DEVICES=2 python3 main.py --action=${action} --dataset=${dataset} --split=${split}

# eval
# CUDA_VISIBLE_DEVICES=2 python3 eval.py --dataset=${dataset} --split=${split}  > "$LOGFILE" 2>&1 &
