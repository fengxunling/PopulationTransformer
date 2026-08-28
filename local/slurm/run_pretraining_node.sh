#!/usr/bin/env bash
# 在计算节点上执行预训练（由 tmux 里的 salloc + srun 调起，见 local/README 或对话记录）。
# 单独拆出来是因为 tmux send-keys 里嵌套引号容易出错。
set -euo pipefail

source /HWFS/yuezhifeng_lab/intern08/miniforge3/etc/profile.d/conda.sh
conda activate wenxin

# num_workers=16，主进程不要再抢 OMP 线程
export OMP_NUM_THREADS=1

cd /HWFS/yuezhifeng_lab/intern08/bci_motor/reference_repo/PopulationTransformer
mkdir -p local/slurm/logs
LOG="local/slurm/logs/pretrain_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-nojob}.log"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-?}  log=${LOG}"
nvidia-smi -L
echo "---- start $(date) ----"

./local/2_run_pretraining.sh 2>&1 | tee "${LOG}"
