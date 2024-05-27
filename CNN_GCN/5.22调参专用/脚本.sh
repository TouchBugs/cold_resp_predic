#!/bin/bash

# /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/归一化数据集.py
# echo wait1
# wait
# /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/创建Data数据32.py
# echo wait2
# wait

nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/5.22调参专用/b32训练网络dp0.2bnB.py > dp0.2bnB 2>&1 &
nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/5.22调参专用/b32训练网络dp0.2bnS.py > dp0.2bnS 2>&1 &
nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/5.22调参专用/b32训练网络dp0.5bnB.py > dp0.5bnB 2>&1 &
nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/5.22调参专用/b32训练网络dp0.5bnS.py > dp0.5bnS 2>&1 &
nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/5.22调参专用/b32训练网络dp0.7bnB.py > dp0.7bnB 2>&1 &
nohup /Data4/gly_wkdir/environment/DeepLpy3.9/bin/python /Data4/gly_wkdir/coldgenepredict/raw_sec/S_italica/CNN_GCN/5.22调参专用/b32训练网络dp0.7bnS.py > dp0.7bnS 2>&1 &

wait

