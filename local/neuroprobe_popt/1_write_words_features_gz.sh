#!/usr/bin/env bash
# 全局 z-score 版提特征。和 1_write_words_features.sh 并列，互不影响：
# 输出目录带 gz_ 前缀，现有的 np_words_d-2.25_t5.0 一个字节都不碰。
#
#   SUBJECT_ID=1 TRIAL_ID=1 ./local/neuroprobe_popt/1_write_words_features_gz.sh
#
# 内存：和 per-window 版一样，实测 MaxRSS 173G（sub_1, 88 电极），要 300G 起。
source "$(dirname "$0")/../_common.sh"

# 窗口和 per-window 版保持一致，只有归一化方式不同 —— 这样两边可以直接对比
WIN_DELTA="${WIN_DELTA:--2.25}"
WIN_DURATION="${WIN_DURATION:-5.0}"
WIN_TAG="${WIN_TAG:-gz_d${WIN_DELTA}_t${WIN_DURATION}}"

FEAT_ROOT="${REPO_DIR}/saved_examples/np_words_${WIN_TAG}"
ELEC_JSON="${REPO_DIR}/local/neuroprobe_popt/electrodes_lite_laplacian.json"

if [ ! -f "${ELEC_JSON}" ]; then
    python3 local/neuroprobe_popt/make_electrode_selection.py \
        --repo-dir "${REPO_DIR}" --out "${ELEC_JSON}"
fi

SUBJECT="sub_${SUBJECT_ID:-1}"
TRIAL=$(printf "trial%03d" "${TRIAL_ID:-1}")
OUT_DIR="${FEAT_ROOT}/${SUBJECT}_${TRIAL}"

if [ -f "${OUT_DIR}/manifest.tsv" ]; then
    echo "=== ${SUBJECT} ${TRIAL} 已存在，跳过（要重提先删 ${OUT_DIR}）"
    exit 0
fi

RUNS_JSON=$(mktemp)
printf '{"%s": ["%s"]}' "${SUBJECT}" "${TRIAL}" > "${RUNS_JSON}"

echo "=== 全局 z-score 提特征 ${SUBJECT} ${TRIAL} -> ${OUT_DIR}"
python3 -m local.neuroprobe_popt.write_words_features_gz \
    --config-path "${REPO_DIR}/conf" \
    +data_prep=pretrain_multi_subj_multi_chan_template \
    ++data_prep.task_name=all_words \
    ++data_prep.brain_runs="${RUNS_JSON}" \
    ++data_prep.electrodes="${ELEC_JSON}" \
    ++data_prep.output_directory="${OUT_DIR}" \
    +preprocessor=multi_elec_spec_pretrained \
    ++preprocessor.upstream_ckpt="${WEIGHTS_DIR}/stft_large_pretrained.pth" \
    +data=subject_data_template \
    ++data.cached_transcript_aligns="${REPO_DIR}/semantics/saved_aligns" \
    ++data.raw_brain_data_dir="${BRAINTREEBANK_DIR}/" \
    ++data.movie_transcripts_dir="${BRAINTREEBANK_DIR}/transcripts" \
    ++data.delta="${WIN_DELTA}" \
    ++data.duration="${WIN_DURATION}"

rm -f "${RUNS_JSON}"
