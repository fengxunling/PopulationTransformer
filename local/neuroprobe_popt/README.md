# neuroprobe_popt —— 在 Neuroprobe 的划分上评 PopT

目的：拿到能和 [neuroprobe.dev](https://neuroprobe.dev) leaderboard 直接对比的数字，
15 个任务 × 12 个 session 全铺开。

## 为什么不能直接用 local/5_run_finetuning.sh 的结果

`5_` 用的是 PopT 自己的划分：`tasks/utils.py` 里的
`train_test_split(all_idxs, test_size=..., random_state=42)`，**随机**划分。同一个
trial 里时间上相邻的词会同时落进 train 和 test，神经信号高度相关 → 结果虚高。

Neuroprobe 用 `KFold(n_splits=k, shuffle=False)`，源码注释：
`# shuffle=False is important to avoid correlated train/test splits!`

实测差距（sub_1 trial1，91 电极，`popt_brainbert_stft` 权重）：

| 任务 | PopT 随机划分 | leaderboard（PopT，多 session 平均） |
| --- | --- | --- |
| volume / rms | 0.933 | 0.766 |
| pitch | 0.912 | 0.596 |

## 为什么要「全词特征」这一步

upstream 的 `data/write_multi_subject_multi_channel.py:50-71` 只给**当前任务用到的
那部分词**写特征：`rms`/`pitch` 只写上下四分位（约一半的词），`pos` 只写名词动词。
Neuroprobe 每个任务的四分位落在不同的词上，逐任务提特征等于重复 15 遍，而且按
onset 时间对齐时会大面积对不上。

所以改成：**每个 session 提一次全部词的特征，15 个任务共用**，只换标签和 fold 索引。
`saved_examples/np_words/<subject>_<trial>/`。

## 用法

```bash
# 0. 提特征：每个 session 一次，之后所有任务共用。需要 GPU。
./local/neuroprobe_popt/1_write_words_features.sh --all        # 12 个 session
SUBJECT_ID=1 TRIAL_ID=1 ./local/neuroprobe_popt/1_write_words_features.sh   # 单个

# 1. 铺开跑：任务 × session × fold，跑完自动汇总成 leaderboard 格式
./local/neuroprobe_popt/run_all_tasks.sh
TASKS="volume pitch" SESSIONS="1:1" ./local/neuroprobe_popt/run_all_tasks.sh
SPLIT=cross_session ./local/neuroprobe_popt/run_all_tasks.sh

# 2. 只重新汇总已有结果
python3 local/neuroprobe_popt/to_leaderboard_json.py --repo-dir . --split within_session
```

`run_all_tasks.sh` 对已有 `results.json` 的格子会跳过，中断了直接重跑就行。

产物：`local/neuroprobe_popt/leaderboard/Within-Session/population_<TASK>.json`，
结构按 `submit_neuroprobe/SUBMIT.md` 的「Formatting results」写。

单任务调试还是用 `run_eval.sh` + `collect_results.py`（走同一套 build 脚本）。

## 关键设计

1. **窗口 = onset ~ onset+1s**，`++data.delta=0.0 ++data.duration=1.0`。leaderboard
   规定的 bin 是 0.0–1.0s。之前用 `conf/data/subject_data_template.yaml` 的默认值
   `delta=-2.5, duration=5.0`，而 BrainBERT 只取频谱中间 10 帧
   （`nperseg=400, noverlap=350` → hop 24ms，约窗口中心 ±122ms），也就是把中心对在
   了 onset 上、**看到了 onset 之前的数据**。改成 0.0/1.0 之后中心落在 onset+0.5s。
2. **电极 = `clean_laplacian` ∩ Neuroprobe lite**（`electrodes_lite_laplacian.json`，
   `make_electrode_selection.py` 生成，82–104 个/被试）。Neuroprobe 规定用 lite 那
   120 个左右，但 PopT 的 laplacian re-reference 要求电极在同一根杆上前后都有邻居
   （`data/h5_data_reader.py:52` 的 `assert len(nbrs)==2`），所以只能取交集。
3. **一个 session 一个输出目录**。upstream 在 subject 循环里只写最后一个 trial 的
   labels（`write_multi_subject_multi_channel.py:370`），一次写多个 trial 的话续跑会
   读到错的 labels，所以 `1_write_words_features.sh` 拆成一个 session 一次。
4. **零改动 upstream**。`write_words_features.py` 是 import 进来给
   `get_subject_data` / `get_raw_data_and_labels` 打补丁，加一个
   `task_name="all_words"`，再调 upstream 自己的 hydra main。

## 还差的部分

1. **`onset` / `speech` 两个任务跑不了**。它们的负样本是「非语音时间窗」，取自
   neuroprobe 的 `nonverbal_df`（`datasets.py:165-169`），不在词表里，全词特征覆盖
   不到。要补得再 dump 一份非语音窗口的特征，窗口位置从 `nonverbal_df` 的 `est_idx`
   来。所以现在最多 13/15 个任务。
2. **`train_accuracy` / `train_roc_auc` 填的是 -1**。PopT 的 runner 只把测试集的
   predicts/labels 写进 `results.json`（`runner.py:200-208`），没有训练集那一遍。
   投稿前要给 runner 加一次训练集评估。
3. **Cross-Subject 划分没做**。`generate_splits_cross_subject` 要多个被试的特征拼在
   一起，PopT 的 manifest 是按 subject 分电极表的，需要额外处理。
4. **预处理和 leaderboard 上那条 PopT 不完全一样**。leaderboard 那条写的是
   "Population Transformer (global z-scoring)"，我们这边是 PopT 论文原生的
   laplacian re-reference + 全 trial notch 滤波。差值要按这个来解释。
