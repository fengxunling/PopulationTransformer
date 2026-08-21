#!/usr/bin/env bash
# 在 Neuroprobe 的 train/val/test 划分上评 PopT，结果可以直接跟 leaderboard 比。
# 需要 GPU —— 先按 CLAUDE.md salloc 到 GPU 分区。
#
#   ./local/neuroprobe_popt/run_eval.sh                      # sub_1 trial1 volume
#   EVAL_NAME=pitch POPT_TASK=pitch ./local/neuroprobe_popt/run_eval.sh
#   SUBJECT_ID=3 TRIAL_ID=0 ./local/neuroprobe_popt/run_eval.sh
source "$(dirname "$0")/../_common.sh"

SUBJECT_ID="${SUBJECT_ID:-1}"
TRIAL_ID="${TRIAL_ID:-1}"
EVAL_NAME="${EVAL_NAME:-volume}"
POPT_TASK="${POPT_TASK:-rms}"          # 到 saved_examples/sub_X_<POPT_TASK>_cr 取特征
WEIGHTS="${WEIGHTS:-popt_brainbert_stft}"
OUT_NAME="np_${EVAL_NAME}_sub${SUBJECT_ID}_trial${TRIAL_ID}"

# label2idx 用的是 set() 迭代顺序（datasets/pt_supervised_task_coords.py），
# 不固定 hash seed 的话 True/False -> 0/1 的映射可能在两次运行之间翻转，
# AUROC 会变成 1-AUROC。
export PYTHONHASHSEED=0

python3 local/neuroprobe_popt/build_neuroprobe_splits.py \
    --repo-dir "${REPO_DIR}" \
    --braintreebank-root "${NEUROPROBE_ROOT_DIR}" \
    --subject-id "${SUBJECT_ID}" \
    --trial-id "${TRIAL_ID}" \
    --eval-name "${EVAL_NAME}" \
    --popt-task "${POPT_TASK}" \
    --out-name "${OUT_NAME}"

N_FOLDS=$(ls -d "${REPO_DIR}/saved_data_splits/${OUT_NAME}_fold"* | wc -l)
for ((k=0; k<N_FOLDS; k++)); do
    echo "=== fold ${k}/${N_FOLDS}"
    python3 run_train.py \
        +exp=multi_elec_feature_extract \
        ++exp.runner.results_dir="${REPO_DIR}/outputs/${OUT_NAME}_fold${k}_${WEIGHTS}" \
        ++exp.runner.device=cuda \
        ++exp.runner.save_checkpoints=False \
        ++model.frozen_upstream=False \
        +task=pt_feature_extract_coords \
        +criterion=pt_feature_extract_coords_criterion \
        +preprocessor=empty_preprocessor \
        +data=pt_supervised_task_coords \
        ++data.data_path="${REPO_DIR}/saved_examples/${OUT_NAME}" \
        ++data.saved_data_split="${REPO_DIR}/saved_data_splits/${OUT_NAME}_fold${k}" \
        +model=pt_downstream_model \
        ++model.upstream_path="${WEIGHTS_DIR}/${WEIGHTS}.pth"
done

python3 local/neuroprobe_popt/collect_results.py --repo-dir "${REPO_DIR}" --prefix "${OUT_NAME}"
