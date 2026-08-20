#!/usr/bin/env bash
#SBATCH --job-name=popt-ft
#SBATCH --partition=rtx6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=local/slurm/logs/%x-%j.out
#
# 用法（在仓库根目录下提交，先用 sinfo -s 确认分区有空位）：
#   mkdir -p local/slurm/logs
#   sbatch --export=ALL,TASK=pos,SUBJECT=sub_1 local/slurm/run_finetuning.sbatch
#
# 长任务用 sbatch，别占着交互式 salloc 过夜。

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate wenxin

srun ./local/5_run_finetuning.sh
