#!/usr/bin/env bash
# 在 Neuroprobe 的划分上把 PopT 铺开跑：任务 × session × fold。需要 GPU。
#
# 前置：./local/neuroprobe_popt/1_write_words_features.sh --all 已经跑完
#      （每个 session 只用提一次特征，之后所有任务共用）
#
#   ./local/neuroprobe_popt/run_all_tasks.sh                      # 13 个任务 × 12 session
#   TASKS="volume pitch" SESSIONS="1:1" ./local/neuroprobe_popt/run_all_tasks.sh
#   SPLIT=cross_session ./local/neuroprobe_popt/run_all_tasks.sh
source "$(dirname "$0")/../_common.sh"

# onset / speech 不在这里 —— 它们的负样本是非语音时间窗，全词特征覆盖不到，
# 见 build_neuroprobe_splits.py 顶部 EVAL_NEEDS_NONVERBAL。
TASKS="${TASKS:-volume delta_volume pitch word_index word_gap gpt2_surprisal \
word_head_pos word_part_speech word_length global_flow local_flow \
frame_brightness face_num}"
SESSIONS="${SESSIONS:-1:1 1:2 2:0 2:4 3:0 3:1 4:0 4:1 7:0 7:1 10:0 10:1}"
SPLIT="${SPLIT:-within_session}"
WIN_TAG="${WIN_TAG:-d-2.25_t5.0}"   # 要和提特征时的 WIN_TAG 一致
WEIGHTS="${WEIGHTS:-popt_brainbert_stft}"

# label2idx 用 set() 迭代顺序（datasets/pt_supervised_task_coords.py:69），
# 不固定 hash seed 的话 True/False -> 0/1 可能在两次运行之间翻转，AUROC 变成 1-AUROC。
export PYTHONHASHSEED=0

FAILED=""
for task in ${TASKS}; do
  for s in ${SESSIONS}; do
    SID="${s%%:*}"; TID="${s##*:}"
    # WIN_TAG 必须进名字：saved_examples / saved_data_splits / outputs 三处都按
    # OUT_NAME 命名，不带窗口标记的话不同窗口配置会共用同一个 outputs 目录，
    # 下面「已有 results.json 就跳过」的逻辑会静默拿旧窗口的结果冒充新的。
    OUT_NAME="np_${SPLIT}_${task}_sub${SID}_trial${TID}_${WIN_TAG}"

    echo ""
    echo "############ ${task}  sub_${SID} trial${TID}  (${SPLIT})"
    if ! python3 local/neuroprobe_popt/build_neuroprobe_splits.py \
            --repo-dir "${REPO_DIR}" \
            --braintreebank-root "${NEUROPROBE_ROOT_DIR}" \
            --subject-id "${SID}" --trial-id "${TID}" \
            --eval-name "${task}" --split "${SPLIT}" --out-name "${OUT_NAME}" \
            --win-tag "${WIN_TAG}"; then
        echo "!!! build 失败，跳过 ${OUT_NAME}"
        FAILED="${FAILED} ${OUT_NAME}:build"
        continue
    fi

    N_FOLDS=$(ls -d "${REPO_DIR}/saved_data_splits/${OUT_NAME}_fold"* 2>/dev/null | wc -l)
    for ((k=0; k<N_FOLDS; k++)); do
        RES_DIR="${REPO_DIR}/outputs/${OUT_NAME}_fold${k}_${WEIGHTS}"
        if [ -f "${RES_DIR}/results.json" ]; then
            echo "=== fold ${k} 已有结果，跳过"
            continue
        fi
        echo "=== fold ${k}/${N_FOLDS}"
        # hydra 默认按 outputs/<日期>/<时:分:秒> 建目录，多作业并发同一秒起步会撞车
        if ! python3 run_train.py \
                hydra.run.dir="${REPO_DIR}/outputs/hydra/${SLURM_JOB_ID:-local}/${OUT_NAME}_fold${k}" \
                +exp=multi_elec_feature_extract \
                ++exp.runner.results_dir="${RES_DIR}" \
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
                ++model.upstream_path="${WEIGHTS_DIR}/${WEIGHTS}.pth"; then
            echo "!!! 训练失败: ${OUT_NAME} fold ${k}"
            FAILED="${FAILED} ${OUT_NAME}:fold${k}"
        fi
    done
  done
done

echo ""
echo "########## 汇总"
python3 local/neuroprobe_popt/to_leaderboard_json.py \
    --repo-dir "${REPO_DIR}" --split "${SPLIT}" --weights "${WEIGHTS}"

if [ -n "${FAILED}" ]; then
    echo ""
    echo "以下没跑成功（汇总里会缺这些格子）:"
    for f in ${FAILED}; do echo "  ${f}"; done
fi
