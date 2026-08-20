#!/usr/bin/env bash
# 写预训练数据（NSP）。对应 upstream 的 0_write_pretraining_data.sh
source "$(dirname "$0")/_common.sh"

# XXX 这里用的是原始数据盘，而 3_write_finetuning_data.sh 用的是组装好的
# ${BRAINTREEBANK_DIR}。原始盘的 h5 是嵌套一层的（sub_1_trial000.h5/sub_1_trial000.h5），
# 且没有 transcripts/，预训练流程跑通前先确认这个路径对不对。
BT_DIR="${BRAINTREEBANK_RAW_DIR}"

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
