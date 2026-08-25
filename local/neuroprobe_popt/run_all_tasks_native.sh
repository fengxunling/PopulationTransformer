#!/usr/bin/env bash
# quickstart 路径：120 个 lite 电极、neuroprobe 原生数据加载（无 notch / 无 laplacian）。
# 提一次特征，15 个任务共用，然后逐任务 build + 训练。
#
#   SESSIONS="1:1" ./local/neuroprobe_popt/run_all_tasks_native.sh
source "$(dirname "$0")/../_common.sh"

TASKS="${TASKS:-onset speech volume delta_volume pitch word_index word_gap \
gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow \
frame_brightness face_num}"
SESSIONS="${SESSIONS:-1:1}"
SPLIT="${SPLIT:-within_session}"
WEIGHTS="${WEIGHTS:-popt_brainbert_stft}"
BEFORE="${BEFORE:-2.25}"
AFTER="${AFTER:-2.75}"
FEAT_ROOT="${REPO_DIR}/saved_examples/np_native_b${BEFORE}_a${AFTER}"

# label2idx 用 set() 迭代顺序，不固定 hash seed 的话 True/False -> 0/1 可能翻转
export PYTHONHASHSEED=0

FAILED=""
for s in ${SESSIONS}; do
  SID="${s%%:*}"; TID="${s##*:}"
  FEAT_DIR="${FEAT_ROOT}/sub_${SID}_trial$(printf '%03d' "${TID}")"

  echo ""
  echo "########## 提特征 sub_${SID} trial${TID}（15 任务共用）"
  python3 local/neuroprobe_popt/write_neuroprobe_features.py \
      --repo-dir "${REPO_DIR}" \
      --braintreebank-root "${NEUROPROBE_ROOT_DIR}" \
      --subject-id "${SID}" --trial-id "${TID}" \
      --before "${BEFORE}" --after "${AFTER}" \
      --out-root "${FEAT_ROOT}" || { echo "!!! 提特征失败"; continue; }

  for task in ${TASKS}; do
    OUT_NAME="npn_${SPLIT}_${task}_sub${SID}_trial${TID}_b${BEFORE}_a${AFTER}"
    echo ""
    echo "########## ${task}  sub_${SID} trial${TID}"
    if ! python3 local/neuroprobe_popt/build_splits_native.py \
            --repo-dir "${REPO_DIR}" \
            --braintreebank-root "${NEUROPROBE_ROOT_DIR}" \
            --feat-dir "${FEAT_DIR}" \
            --subject-id "${SID}" --trial-id "${TID}" \
            --eval-name "${task}" --split "${SPLIT}" --out-name "${OUT_NAME}"; then
        echo "!!! build 失败: ${OUT_NAME}"; FAILED="${FAILED} ${task}:build"; continue
    fi

    N_FOLDS=$(ls -d "${REPO_DIR}/saved_data_splits/${OUT_NAME}_fold"* 2>/dev/null | wc -l)
    for ((k=0; k<N_FOLDS; k++)); do
        RES_DIR="${REPO_DIR}/outputs/${OUT_NAME}_fold${k}_${WEIGHTS}"
        [ -f "${RES_DIR}/results.json" ] && { echo "=== fold ${k} 已有结果，跳过"; continue; }
        echo "=== fold ${k}/${N_FOLDS}"
        # hydra 默认按 outputs/<日期>/<时:分:秒> 建目录，多个作业并发时同一秒起步会撞车
        python3 run_train.py \
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
            ++model.upstream_path="${WEIGHTS_DIR}/${WEIGHTS}.pth" \
          || { echo "!!! 训练失败: ${OUT_NAME} fold ${k}"; FAILED="${FAILED} ${task}:fold${k}"; }
    done
  done
done

[ -n "${FAILED}" ] && { echo ""; echo "没跑成功的:"; for f in ${FAILED}; do echo "  ${f}"; done; }
echo "@@@@@ NATIVE DONE"
