#!/usr/bin/env bash
# 写微调数据。对应 upstream 的 3_write_finetuning_data.sh
# 覆盖 TASK： TASK=pitch ./local/3_write_finetuning_data.sh
source "$(dirname "$0")/_common.sh"

# 输出目录必须带上 TASK：write_trial_data 会跳过已存在的 subject/trial 目录，
# 复用别的 task 的目录会静默拿到那个 task 缓存下来的 npy。
TASK="${TASK:-pos}"

python3 -m data.write_multi_subject_multi_channel \
+data_prep=pretrain_multi_subj_multi_chan_template \
++data_prep.task_name=${TASK} \
++data_prep.brain_runs=${REPO_DIR}/trial_selections/test_trials.json \
++data_prep.electrodes=${REPO_DIR}/electrode_selections/clean_laplacian.json \
++data_prep.output_directory=${REPO_DIR}/saved_examples/all_test_${TASK} \
+preprocessor=multi_elec_spec_pretrained \
++preprocessor.upstream_ckpt=${WEIGHTS_DIR}/stft_large_pretrained.pth \
+data=subject_data_template \
++data.cached_transcript_aligns=${REPO_DIR}/semantics/saved_aligns \
++data.raw_brain_data_dir=${BRAINTREEBANK_DIR}/ \
++data.movie_transcripts_dir=${BRAINTREEBANK_DIR}/transcripts

# 去掉了 upstream 的 ++data.cached_data_array=${REPO_DIR}/cached_data_arrays/
# —— 缓存的 array 不带 task 信息，换 task 时会串味。
