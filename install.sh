#!/bin/bash

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir torch==1.12.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir tensorflow==2.13.1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir scipy scikit-learn tqdm logzero pandas seaborn numba paramiko jupyterlab pettingzoo
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir gymnasium==1.1.1  # for tianshou's new version
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir numpy==1.22.4 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --no-cache-dir pandas==1.5.3 


#pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade torch

#pip install -i https://mirrors.aliyun.com/pypi/simple/  --upgrade --no-cache-dir tensorflow

