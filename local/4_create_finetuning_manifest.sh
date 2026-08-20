#!/usr/bin/env bash
# 生成微调 manifest。对应 upstream 的 4_create_finetuning_manifest.sh
# 覆盖： SUBJECT=sub_2 TASK=pos ./local/4_create_finetuning_manifest.sh
source "$(dirname "$0")/_common.sh"

SUBJECT="${SUBJECT:-sub_1}"
TASK="${TASK:-pitch}"

python3 -m data.make_subject_specific_manifest \
+data_prep=subject_specific_manifest \
++data_prep.data_path=${REPO_DIR}/saved_examples/all_test_${TASK} \
++data_prep.subj=${SUBJECT} \
++data_prep.out_path=${REPO_DIR}/saved_examples/${SUBJECT}_${TASK}_cr
