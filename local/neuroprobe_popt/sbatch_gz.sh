#!/usr/bin/env bash
#SBATCH --job-name=gz
#SBATCH --partition=rtxp6000bws,rtx5090s,rtx6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --output=logs/%x_%j.log
# 全局 z-score 路径：提特征 + 跑 13 个任务，一个作业一个 session。
#   SESSIONS="1:1" sbatch --export=ALL,SESSIONS local/neuroprobe_popt/sbatch_gz.sh
source /HWFS/yuezhifeng_lab/intern08/miniforge3/etc/profile.d/conda.sh
conda activate wenxin
cd /HWFS/yuezhifeng_lab/intern08/bci_motor/reference_repo/PopulationTransformer

export WIN_TAG="${WIN_TAG:-gz_d-2.25_t5.0}"
export SESSIONS="${SESSIONS:-1:1}"
S="${SESSIONS%% *}"
export SUBJECT_ID="${S%%:*}"
export TRIAL_ID="${S##*:}"

WIN_DELTA="${WIN_DELTA:--2.25}" WIN_DURATION="${WIN_DURATION:-5.0}" WIN_TAG="${WIN_TAG}" \
  ./local/neuroprobe_popt/1_write_words_features_gz.sh || { echo "!!! 提特征失败"; exit 1; }

# run_all_tasks.sh 用 WIN_TAG 定位特征目录和 OUT_NAME，所以自动就跟 per-window 那批分开
./local/neuroprobe_popt/run_all_tasks.sh
echo "@@@@@ GZ DONE"
