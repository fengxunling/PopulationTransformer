#!/usr/bin/env bash
# 给一个 Neuroprobe session 写「全部词」的 PopT/BrainBERT 特征。
# 15 个任务共用这一份，所以每个 session 只用跑一次。需要 GPU。
#
#   SUBJECT_ID=1 TRIAL_ID=1 ./local/neuroprobe_popt/1_write_words_features.sh
#   ./local/neuroprobe_popt/1_write_words_features.sh --all      # 12 个 session 全跑
#
# 内存要给够：全词一次性加载是 [n_elec, n_words, 2048] 的 float64，sub_1 约 15GB，
# 再叠上整段 trial 的原始数据和 laplacian 的邻居数组。salloc -c 8 (~48GB) 会 OOM。
#   salloc -p rtxp6000bws --gres=gpu:1 -c 32 --mem=400G
source "$(dirname "$0")/../_common.sh"

# 窗口：BrainBERT 只取频谱中间 10 帧（preprocessors/multi_elec_spec_pretrained.py:61），
# 所以「模型实际读到的」是 onset+DELTA+DURATION/2 前后各约 122ms。
# DURATION 保持 5.0 匹配预训练（conf/data/pretraining_subject_data_template.yaml:11），
# 靠 DELTA 把那 10 帧挪进 leaderboard 规定的 [0, 1s] 里：
#   -2.25 -> 中心 +0.25s, 覆盖 0.13~0.37s（推荐，落在听觉响应上）
#   -2.0  -> 中心 +0.50s, 覆盖 0.38~0.62s
#   -2.5  -> 中心 onset，跨过 onset，不合规
WIN_DELTA="${WIN_DELTA:--2.25}"
WIN_DURATION="${WIN_DURATION:-5.0}"
# 输出目录带上窗口标记，不同窗口的特征不会互相覆盖
WIN_TAG="${WIN_TAG:-d${WIN_DELTA}_t${WIN_DURATION}}"

FEAT_ROOT="${REPO_DIR}/saved_examples/np_words_${WIN_TAG}"
ELEC_JSON="${REPO_DIR}/local/neuroprobe_popt/electrodes_lite_laplacian.json"

if [ ! -f "${ELEC_JSON}" ]; then
    python3 local/neuroprobe_popt/make_electrode_selection.py \
        --repo-dir "${REPO_DIR}" --out "${ELEC_JSON}"
fi

# Neuroprobe lite 的 12 个 session（neuroprobe.config.NEUROPROBE_LITE_SUBJECT_TRIALS）
ALL_SESSIONS="1:1 1:2 2:0 2:4 3:0 3:1 4:0 4:1 7:0 7:1 10:0 10:1"

if [ "${1:-}" = "--all" ]; then
    SESSIONS="${ALL_SESSIONS}"
else
    SESSIONS="${SUBJECT_ID:-1}:${TRIAL_ID:-1}"
fi

for s in ${SESSIONS}; do
    SID="${s%%:*}"; TID="${s##*:}"
    SUBJECT="sub_${SID}"
    TRIAL=$(printf "trial%03d" "${TID}")
    OUT_DIR="${FEAT_ROOT}/${SUBJECT}_${TRIAL}"

    if [ -f "${OUT_DIR}/manifest.tsv" ]; then
        echo "=== ${SUBJECT} ${TRIAL} 已存在，跳过（要重提先删 ${OUT_DIR}）"
        continue
    fi

    # 一个 session 一个输出目录。upstream 的 write_labels 在 subject 循环里只写
    # 最后一个 trial 的 labels（data/write_multi_subject_multi_channel.py:370），
    # 一次写多个 trial 的话续跑会读到错的 labels，所以拆开跑。
    RUNS_JSON=$(mktemp)
    printf '{"%s": ["%s"]}' "${SUBJECT}" "${TRIAL}" > "${RUNS_JSON}"

    echo "=== 写 ${SUBJECT} ${TRIAL} 的全词特征 -> ${OUT_DIR}"
    # --config-path 必须显式给：hydra 的 config_path 是相对「调用 main() 的那个
    # 文件」算的，入口挪到 local/neuroprobe_popt/ 之后 upstream 写的 "../conf"
    # 会指到 local/conf。
    python3 -m local.neuroprobe_popt.write_words_features \
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
done
