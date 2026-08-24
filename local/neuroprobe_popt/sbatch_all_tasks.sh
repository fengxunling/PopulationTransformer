#!/usr/bin/env bash
#SBATCH --job-name=np_popt
#SBATCH --partition=rtxp6000bws
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=%x_%j.log
# 铺开跑 Neuroprobe 任务。长任务用 sbatch，别占交互式 allocation。
#   sbatch local/neuroprobe_popt/sbatch_all_tasks.sh
#   SESSIONS="1:1 1:2" sbatch --export=ALL,SESSIONS local/neuroprobe_popt/sbatch_all_tasks.sh
source /HWFS/yuezhifeng_lab/intern08/miniforge3/etc/profile.d/conda.sh
conda activate wenxin
cd /HWFS/yuezhifeng_lab/intern08/bci_motor/reference_repo/PopulationTransformer
export WIN_TAG="${WIN_TAG:-d-2.25_t5.0}"
export SESSIONS="${SESSIONS:-1:1}"
./local/neuroprobe_popt/run_all_tasks.sh
echo "@@@@@ SBATCH DONE"
