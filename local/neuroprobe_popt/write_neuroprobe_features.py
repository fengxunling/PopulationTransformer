"""按 quickstart.ipynb 的方式提 PopT/BrainBERT 特征：数据全部从 neuroprobe 的 API 拿。

跟 write_words_features.py（走 PopT 自己的 SubjectData）的区别：

|        | 本脚本                          | write_words_features.py        |
| ------ | ------------------------------- | ------------------------------ |
| 电极   | 120 个 Neuroprobe lite 全量     | 88 个（clean_laplacian ∩ lite）|
| 预处理 | 原始信号，无 notch、无 re-ref   | notch 滤波 + laplacian re-ref  |
| 窗口   | 直接用 neuroprobe 的 est_idx    | 按词的 onset 时间对齐          |
| 覆盖   | 100%，onset/speech 也能做       | ~95%，onset/speech 做不了      |

窗口长度仍是 5s（BEFORE=2.25s, AFTER=2.75s），中心落在 onset+0.25s，和 PopT 那条
路径的 delta=-2.25/duration=5.0 等价，方便两边直接对比。为什么不用 1s：BrainBERT
只取频谱中间 10 帧，且 STFT 的 zscore 在窗口内做，1s 窗口实测掉到随机水平。

一个 session 的 15 个任务共用一份特征 —— 按窗口起点去重，同一个窗口只提一次。

注意：neuroprobe 返回的是**未滤波**的原始信号，60/120/180 Hz 工频噪声在
BrainBERT 的输入频段（0~205 Hz）内。这是本路径和 PopT 原生管线的主要差异。
"""
import argparse
import csv
import json
import os
import sys

ALL_EVALS = [
    "onset", "speech", "volume", "delta_volume", "pitch",
    "word_index", "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--braintreebank-root", required=True)
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--trial-id", type=int, default=1)
    p.add_argument("--before", type=float, default=2.25, help="onset 前取多少秒")
    p.add_argument("--after", type=float, default=2.75, help="onset 后取多少秒")
    p.add_argument("--out-root", default=None)
    p.add_argument("--upstream-ckpt", default=None)
    p.add_argument("--tasks", default=",".join(ALL_EVALS))
    return p.parse_args()


def main():
    args = parse_args()
    repo = args.repo_dir
    os.environ["ROOT_DIR_BRAINTREEBANK"] = args.braintreebank_root
    sys.path.insert(0, repo)

    import numpy as np
    import torch
    import pandas as pd
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import neuroprobe.config as npcfg
    from neuroprobe import BrainTreebankSubject, BrainTreebankSubjectTrialBenchmarkDataset

    sid, tid = args.subject_id, args.trial_id
    subject_name = "sub_%d" % sid
    before = int(args.before * npcfg.SAMPLING_RATE)
    after = int(args.after * npcfg.SAMPLING_RATE)
    tasks = [t for t in args.tasks.split(",") if t]

    out_root = args.out_root or os.path.join(
        repo, "saved_examples",
        "np_native_b%g_a%g" % (args.before, args.after))
    out_dir = os.path.join(out_root, "%s_trial%03d" % (subject_name, tid))
    feat_dir = os.path.join(out_dir, "feats")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "localization"), exist_ok=True)

    ckpt = args.upstream_ckpt or os.path.join(
        repo, "pretrained_weights", "stft_large_pretrained.pth")

    # ---- 枚举 15 个任务用到的所有窗口，去重 --------------------------------
    # 用 cache=False 的 subject 只做索引枚举，避免触发部分缓存后面还要重缓存
    idx_subject = BrainTreebankSubject(sid, allow_corrupted=False, cache=False,
                                       dtype=torch.float32)
    windows = {}          # window_from -> window_to
    task_meta = {}
    elec_labels = None
    for task in tasks:
        try:
            ds = BrainTreebankSubjectTrialBenchmarkDataset(
                idx_subject, tid, dtype=torch.float32, eval_name=task,
                output_indices=True, output_dict=False,
                start_neural_data_before_word_onset=before,
                end_neural_data_after_word_onset=after, lite=True)
        except Exception as e:
            print("跳过 %s: %s" % (task, e))
            continue
        if elec_labels is None:
            elec_labels = list(ds.electrode_labels)
        n = 0
        for i in range(ds.n_samples):
            (wf, wt), _ = ds[i]
            windows[int(wf)] = int(wt)
            n += 1
        task_meta[task] = n
        print("%-18s %d 个样本" % (task, n))

    if not windows:
        sys.exit("一个任务都没建起来")
    print("\n15 个任务合计 %d 个样本，去重后 %d 个唯一窗口"
          % (sum(task_meta.values()), len(windows)))

    # ---- 电极：lite 全量，但要在 PopT 的 localization 表里找得到坐标 --------
    loc_path = os.path.join(repo, "braintreebank_data", "localization",
                            subject_name, "depth-wm.csv")
    loc_df = pd.read_csv(loc_path).set_index("Electrode", drop=False)
    keep = [(i, e) for i, e in enumerate(elec_labels) if e in loc_df.index]
    missing = [e for e in elec_labels if e not in loc_df.index]
    if missing:
        print("有 %d 个 lite 电极在 localization 表里没有坐标，丢弃: %s"
              % (len(missing), missing))
    keep_idx = [i for i, _ in keep]
    keep_labels = [e for _, e in keep]
    print("电极: lite %d -> 实际用 %d" % (len(elec_labels), len(keep_labels)))

    loc_df.loc[keep_labels].to_csv(
        os.path.join(out_dir, "localization", "%s.csv" % subject_name), index=False)
    with open(os.path.join(out_dir, "all_ordered_electrodes.json"), "w") as fd:
        json.dump({subject_name: keep_labels}, fd)

    # ---- 提特征 ------------------------------------------------------------
    from preprocessors import build_preprocessor
    extracter = build_preprocessor(OmegaConf.create({
        "name": "multi_elec_spec_pretrained", "spec_name": "stft",
        "freq_channel_cutoff": 40, "nperseg": 400, "noverlap": 350,
        "normalizing": "zscore", "upstream_ckpt": ckpt,
    }))

    subject = BrainTreebankSubject(sid, allow_corrupted=False, cache=True,
                                   dtype=torch.float32)
    subject.load_neural_data(tid)   # 整段缓存，避免逐窗口重缓存

    todo = sorted(windows)
    print("\n开始提特征，共 %d 个窗口 -> %s" % (len(todo), feat_dir))
    for wf in tqdm(todo):
        save_path = os.path.join(feat_dir, "%d.npy" % wf)
        if os.path.exists(save_path):
            continue
        x = subject.get_all_electrode_data(tid, window_from=wf, window_to=windows[wf])
        x = x[keep_idx].numpy()          # [n_elec, n_time]
        np.save(save_path, extracter(x).numpy())

    with open(os.path.join(out_dir, "meta.json"), "w") as fd:
        json.dump({"subject_id": sid, "trial_id": tid,
                   "before_samples": before, "after_samples": after,
                   "n_windows": len(todo), "n_electrodes": len(keep_labels),
                   "electrode_labels": keep_labels,
                   "samples_per_task": task_meta,
                   "source": "neuroprobe native (raw, no notch, no rereference)"}, fd, indent=2)
    print("\n完成: %s" % out_dir)


if __name__ == "__main__":
    main()
