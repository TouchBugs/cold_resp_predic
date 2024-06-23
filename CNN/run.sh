#!/bin/bash

# 设置需要监控的进程ID
TARGET_PID=151278

# 设置待启动的命令
CMD="nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN/GRU.py > nohup.out 2>&1 &"

# 检测目标进程是否存在的函数
is_process_running() {
    ps -p $TARGET_PID > /dev/null 2>&1
}

# 主监控逻辑
while is_process_running; do
    # 进程仍在运行，等待60秒后再次检查
    echo "runing"
    sleep 60
done

# 进程结束，运行目标命令
echo "进程 $TARGET_PID 已结束，启动命令：$CMD"
eval $CMD
