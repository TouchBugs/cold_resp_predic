#!/bin/bash
set -euo pipefail

# 错误处理函数
error_handler() {
    local last_command=$1
    local last_code=$2
    echo "Error: command '${last_command}' failed with exit code ${last_code}."
    echo "See ${log_file} for more details."
    exit 1
}

trap 'error_handler "${BASH_COMMAND}" "$?"' ERR

# 定义参数的推荐数值范围
lrs=(0.001 0.005)
weight_decays=(0.0001 0.001)
freeze_GRUs=(1)
threathholds=(0.4 0.5)
hidden_size2s=(128 256)
hidden_size3s=(64 128)

# 日志记录
log_file="run_script.log"
exec 3>&1 1>>${log_file} 2>&1

log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1"
}

log "Starting parameter search at $(date)"

# 参数检查
if [[ -z "$lrs" || -z "$weight_decays" || -z "$freeze_GRUs" || -z "$threathholds" || -z "$hidden_size2s" || -z "$hidden_size3s" ]]; then
    log "One or more parameter arrays are empty. Please provide valid values."
    exit 1
fi

log "Parameters:"
log "lrs: ${lrs[*]}"
log "weight_decays: ${weight_decays[*]}"
log "freeze_GRUs: ${freeze_GRUs[*]}"
log "threathholds: ${threathholds[*]}"
log "hidden_size2s: ${hidden_size2s[*]}"
log "hidden_size3s: ${hidden_size3s[*]}"

# 遍历所有参数组合
for lr in "${lrs[@]}"; do
  for weight_decay in "${weight_decays[@]}"; do
    for freeze_GRU in "${freeze_GRUs[@]}"; do
      for threathhold in "${threathholds[@]}"; do
        for hidden_size2 in "${hidden_size2s[@]}"; do
          for hidden_size3 in "${hidden_size3s[@]}"; do
            log "Running with lr=$lr, weight_decay=$weight_decay, freeze_GRU=$freeze_GRU, threathhold=$threathhold, hidden_size2=$hidden_size2, hidden_size3=$hidden_size3"
            
            nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/GRU.py \
              -lr "$lr" \
              -weight_decay "$weight_decay" \
              -freeze_GRU "$freeze_GRU" \
              -threathhold "$threathhold" \
              -hidden_size2 "$hidden_size2" \
              -hidden_size3 "$hidden_size3" \
              -epoch 100 \
              > nohup_"${lr}"_"${weight_decay}"_"${freeze_GRU}"_"${threathhold}"_"${hidden_size2}"_"${hidden_size3}".out 2>&1 &
            
            pid=$!
            log "Started process with PID $pid"

            # 等待任务完成
            wait "$pid"
            if [ $? -eq 0 ]; then
                log "Process $pid completed successfully."
            else
                log "Process $pid failed."
                exit 1
            fi
          done
        done
      done
    done
  done
done

log "Parameter search finished at $(date)"
exec 1>&3 3>&-
