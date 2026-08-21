#!/usr/bin/env bash
# 跑微调 / 特征提取。需要 GPU —— 先 salloc 到 GPU 分区再执行。
# 对应 upstream 的 5_run_finetuning.sh
# 覆盖： SUBJECT=sub_2 TASK=pos ./local/5_run_finetuning.sh
source "$(dirname "$0")/_common.sh"

SUBJECT="${SUBJECT:-sub_1}"
TASK="${TASK:-pitch}"
N="${N:-1}"
NAME="${NAME:-popt_brainbert_stft}"
WEIGHTS="${WEIGHTS:-popt_brainbert_stft}"

python3 run_train.py \
+exp=multi_elec_feature_extract \
++exp.runner.results_dir=${REPO_DIR}/outputs/${SUBJECT}_${TASK}_top${N}_${NAME} \
++exp.runner.device=cuda \
+task=pt_feature_extract_coords \
+criterion=pt_feature_extract_coords_criterion \
+preprocessor=empty_preprocessor \
+data=pt_supervised_task_coords \
++data.data_path=${REPO_DIR}/saved_examples/${SUBJECT}_${TASK}_cr \
++data.saved_data_split=${REPO_DIR}/saved_data_splits/${SUBJECT}_${TASK}_fine_tuning \
+model=pt_downstream_model \
++model.upstream_path=${WEIGHTS_DIR}/${WEIGHTS}.pth

# 去掉了 upstream 的
#   ++data.sub_sample_electrodes=${REPO_DIR}/electrode_selections/debug_electrodes.json
# —— 那是 debug 用的小电极子集，跑完整实验时不要加。
