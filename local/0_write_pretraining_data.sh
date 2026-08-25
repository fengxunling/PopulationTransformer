#!/usr/bin/env bash
# 写预训练数据（NSP）。对应 upstream 的 0_write_pretraining_data.sh
source "$(dirname "$0")/_common.sh"

# 必须用组装好的 ${BRAINTREEBANK_DIR}，不能用原始盘：
#   - 原始盘没有 electrode_labels/，data/h5_data.py:29-30 的 glob 会匹配到 0 个，
#     assert len(electrode_labels_file)==1 直接 AssertionError
#   - 原始盘的 h5 还嵌套一层（sub_1_trial000.h5/sub_1_trial000.h5）
# braintreebank_data/ 里 metadata 是实体文件、all_subject_data/ 下的 h5 是指向
# 原始盘的符号链接，两个问题都绕开了。3_write_finetuning_data.sh 用的也是它。
BT_DIR="${BRAINTREEBANK_DIR}"

python3 -m data.write_nsp_pretraining_data \
+preprocessor=multi_elec_spec_pretrained \
++preprocessor.upstream_ckpt=${WEIGHTS_DIR}/stft_large_pretrained.pth \
+data_prep=pretrain_multi_subj_multi_chan_template \
++data_prep.task_name=nsp_pretraining \
++data_prep.brain_runs=${REPO_DIR}/trial_selections/pretrain_split_trials.json \
++data_prep.electrodes=${REPO_DIR}/electrode_selections/clean_laplacian.json \
++data_prep.output_directory=${REPO_DIR}/saved_examples/cr_pretrain_examples \
+data=pretraining_subject_data_template \
++data.cached_transcript_aligns=${REPO_DIR}/semantics/saved_aligns \
++data.cached_data_array=${REPO_DIR}/cached_data_arrays/ \
++data.raw_brain_data_dir=${BT_DIR}
