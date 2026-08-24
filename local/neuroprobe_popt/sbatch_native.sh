#!/usr/bin/env bash
#SBATCH --job-name=np_native
#SBATCH --partition=rtxp6000bws
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --output=%x_%j.log
source /HWFS/yuezhifeng_lab/intern08/miniforge3/etc/profile.d/conda.sh
conda activate wenxin
cd /HWFS/yuezhifeng_lab/intern08/bci_motor/reference_repo/PopulationTransformer
export SESSIONS="${SESSIONS:-1:1}"
./local/neuroprobe_popt/run_all_tasks_native.sh
