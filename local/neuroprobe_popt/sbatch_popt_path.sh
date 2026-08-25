#!/usr/bin/env bash
#SBATCH --job-name=popt_path
#SBATCH --partition=rtxp6000bws,rtx5090s,rtx6000ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --output=logs/%x_%j.log
# PopT 原生路径（88 电极 = clean_laplacian ∩ lite，notch 滤波 + laplacian re-ref）。
# 提特征 + 跑 13 个任务，一个作业一个 session。
#
# 内存：实测提特征 MaxRSS 173G（sub_1, 88 电极），48G 会 OOM。电极更多的被试
# （sub_7 104 个）按比例放大，300G 留了余量。
#
#   SESSIONS="1:2" sbatch --export=ALL,SESSIONS local/neuroprobe_popt/sbatch_popt_path.sh
source /HWFS/yuezhifeng_lab/intern08/miniforge3/etc/profile.d/conda.sh
conda activate wenxin
cd /HWFS/yuezhifeng_lab/intern08/bci_motor/reference_repo/PopulationTransformer

export WIN_TAG="${WIN_TAG:-d-2.25_t5.0}"
export SESSIONS="${SESSIONS:-1:1}"
S="${SESSIONS%% *}"
export SUBJECT_ID="${S%%:*}"
export TRIAL_ID="${S##*:}"

# 1. 提特征（13 个任务共用；WIN_DELTA/WIN_DURATION 要和 WIN_TAG 对应）
WIN_DELTA=-2.25 WIN_DURATION=5.0 WIN_TAG="${WIN_TAG}" \
  ./local/neuroprobe_popt/1_write_words_features.sh || { echo "!!! 提特征失败"; exit 1; }

# 2. 跑 13 个任务（onset/speech 这条路径做不了）
./local/neuroprobe_popt/run_all_tasks.sh
echo "@@@@@ POPT-PATH DONE"
