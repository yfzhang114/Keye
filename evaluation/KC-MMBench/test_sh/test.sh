
# 获取时间戳
timestamp=$(date +%Y%m%d-%H:%M:%S)
echo "Now time: $timestamp" 

log_dir="./log"

# log_dir 不存在就创建
if [ ! -d "$log_dir" ]; then
    echo "log_dir not found, create it"
    mkdir -p $log_dir
fi

torchrun --master-port 1241 --nproc-per-node=8 run.py \
    --config test_sh/cfg_test_single.json \
    --mode all  \
    2>&1 | tee $log_dir/$timestamp.log

