"""按 Neuroprobe 的任务标签 + 官方划分，从「全词特征」组装 PopT 能吃的 manifest。

流程：
  1_write_words_features.sh 已经为每个 session 写好了全部词的 BrainBERT 特征
  （saved_examples/np_words/<subject>_<trial>/，窗口 = onset ~ onset+1s）。
  这里按词的 onset 时间把 Neuroprobe 的样本对到那些特征上，然后：
    - 用 Neuroprobe 的标签重写 labels.tsv；
    - 把 Neuroprobe 的 fold 索引写成 splits.json（tasks/utils.py:35-49 会优先读它，
      而不是自己 random_state=42 随机划分）。

不重提特征 —— 15 个任务共用同一份，这是能一天跑完 180 个格子的关键。

支持 within_session（2 fold）和 cross_session（1 fold，训练集来自同被试的另一个 trial）。
onset / speech 两个任务暂不支持，见下面 EVAL_NEEDS_NONVERBAL 的说明。
"""
import argparse
import csv
import json
import os
import shutil
import sys

# 这两个任务的负样本是「非语音时间窗」，取自 neuroprobe 的 nonverbal_df
# （datasets.py:165-169 + _positive_negative_getitem__ 的 nonverbal 分支），
# 不在词表里，所以全词特征覆盖不到。要支持得再 dump 一份非语音窗口的特征。
EVAL_NEEDS_NONVERBAL = ("onset", "speech")

ALL_EVAL_NAMES = [
    "volume", "delta_volume", "pitch",
    "word_index", "word_gap", "gpt2_surprisal", "word_head_pos",
    "word_part_speech", "word_length",
    "global_flow", "local_flow", "frame_brightness", "face_num",
    "onset", "speech",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--braintreebank-root", required=True,
                   help="neuroprobe 用的数据根目录（h5 在根下、metadata 不嵌套的那份）")
    p.add_argument("--features-root", default=None,
                   help="全词特征根目录，默认 <repo>/saved_examples/np_words_<WIN_TAG>")
    p.add_argument("--win-tag", default="d-2.25_t5.0",
                   help="窗口标记，要和 1_write_words_features.sh 的 WIN_TAG 一致")
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--trial-id", type=int, default=1, help="测试用的 trial")
    p.add_argument("--eval-name", default="volume", help="neuroprobe 任务名")
    p.add_argument("--split", default="within_session",
                   choices=["within_session", "cross_session"])
    p.add_argument("--out-name", default=None)
    p.add_argument("--window-samples", type=int, default=None,
                   help="neuroprobe 窗口长度，默认 1s = SAMPLING_RATE")
    p.add_argument("--min-coverage", type=float, default=0.9,
                   help="对齐率低于这个值就报错退出，防止静默拿到半份数据")
    return p.parse_args()


class SessionFeats(object):
    """一个 session 的全词特征：onset 时间 -> manifest 行。"""

    def __init__(self, features_root, subject_name, trial_id):
        trial_tag = "trial%03d" % trial_id
        self.dir = os.path.join(features_root, "%s_%s" % (subject_name, trial_tag))
        if not os.path.isfile(os.path.join(self.dir, "manifest.tsv")):
            sys.exit("找不到 %s/manifest.tsv，先跑 "
                     "SUBJECT_ID=%s TRIAL_ID=%d ./local/neuroprobe_popt/1_write_words_features.sh"
                     % (self.dir, subject_name.replace("sub_", ""), trial_id))

        with open(os.path.join(self.dir, "manifest.tsv")) as fd:
            self.manifest = [tuple(r) for r in csv.reader(fd, delimiter="\t")]
        with open(os.path.join(self.dir, "labels.tsv")) as fd:
            rows = [tuple(r) for r in csv.reader(fd, delimiter="\t")]
        assert len(self.manifest) == len(rows), \
            "%s 的 manifest 和 labels 行数不一致" % self.dir

        # 第 1 列是词的 onset 时间（write_words_features.py 写的）
        self.start2row = {}
        for i, row in enumerate(rows):
            self.start2row.setdefault(round(float(row[1]), 4), i)


def enumerate_samples(inner):
    """复现 neuroprobe datasets.py 的 _positive_negative_getitem__ 索引逻辑。

    返回 [(neuroprobe 样本索引, 标签, 词的 onset 时间)]。
    """
    out = []
    for idx in range(inner.n_samples):
        label = (idx + 1) % inner.n_classes
        word_index = inner.label_indices[label][idx // inner.n_classes]
        row = inner.all_words_df.iloc[word_index]
        out.append((idx, label, round(float(row["start"]), 4)))
    return out


def unwrap(ds):
    """Subset / Subset(Subset) -> (最内层 dataset, 在最内层里的索引列表)"""
    from torch.utils.data import Subset
    idxs = None
    while isinstance(ds, Subset):
        idxs = ds.indices if idxs is None else [ds.indices[i] for i in idxs]
        ds = ds.dataset
    if idxs is None:
        idxs = list(range(len(ds)))
    return ds, list(idxs)


def main():
    args = parse_args()
    repo = args.repo_dir
    os.environ["ROOT_DIR_BRAINTREEBANK"] = args.braintreebank_root
    features_root = args.features_root or os.path.join(
        repo, "saved_examples", "np_words_%s" % args.win_tag)

    if args.eval_name in EVAL_NEEDS_NONVERBAL:
        sys.exit("%s 的负样本是非语音时间窗，不在全词特征里。"
                 "见 build_neuroprobe_splits.py 顶部 EVAL_NEEDS_NONVERBAL 的说明。"
                 % args.eval_name)

    import torch
    import neuroprobe.config as npcfg
    from neuroprobe import BrainTreebankSubject
    import neuroprobe.train_test_splits as tts

    sid, tid = args.subject_id, args.trial_id
    subject_name = "sub_%d" % sid
    window = args.window_samples or npcfg.SAMPLING_RATE

    subject = BrainTreebankSubject(sid, allow_corrupted=False, cache=False,
                                   dtype=torch.float32)
    ds_kwargs = dict(dtype=torch.float32, output_indices=True, output_dict=False,
                     start_neural_data_before_word_onset=0,
                     end_neural_data_after_word_onset=window, lite=True)
    if args.split == "within_session":
        folds = tts.generate_splits_within_session(subject, tid, args.eval_name, **ds_kwargs)
    else:
        folds = tts.generate_splits_cross_session(subject, tid, args.eval_name, **ds_kwargs)

    # ---- 收集用到的所有 (dataset, trial_id)，一个 trial 加载一份特征 -----------
    inner_datasets = {}   # id(inner) -> (inner, trial_id)
    for fold in folds:
        for key in ("train_dataset", "val_dataset", "test_dataset"):
            inner, _ = unwrap(fold[key])
            inner_datasets[id(inner)] = (inner, inner.trial_id)

    feats = {}            # trial_id -> SessionFeats
    for inner, trial_id in inner_datasets.values():
        if trial_id not in feats:
            feats[trial_id] = SessionFeats(features_root, subject_name, trial_id)

    # ---- 对齐：(dataset, 样本索引) -> 全局行号 --------------------------------
    out_manifest, out_labels = [], []
    sample2row = {}
    n_total = n_matched = 0
    for inner, trial_id in inner_datasets.values():
        sf = feats[trial_id]
        for np_idx, label, start in enumerate_samples(inner):
            n_total += 1
            src_row = sf.start2row.get(start)
            if src_row is None:
                continue          # 这个词 PopT 侧没写出来（转录对齐失败等）
            n_matched += 1
            sample2row[(id(inner), np_idx)] = len(out_manifest)
            out_manifest.append(sf.manifest[src_row])
            out_labels.append((str(bool(label)), start))

    coverage = float(n_matched) / max(n_total, 1)
    print("对齐 %d/%d 个样本 (%.1f%%)，涉及 trial %s"
          % (n_matched, n_total, 100 * coverage, sorted(feats.keys())))
    if coverage < args.min_coverage:
        sys.exit("对齐率 %.1f%% 低于 --min-coverage %.1f%%，先查特征是不是提全了"
                 % (100 * coverage, 100 * args.min_coverage))

    # ---- 写 manifest ---------------------------------------------------------
    out_name = args.out_name or ("np_%s_%s_sub%d_trial%d"
                                 % (args.split, args.eval_name, sid, tid))
    out_dir = os.path.join(repo, "saved_examples", out_name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "localization"))

    with open(os.path.join(out_dir, "manifest.tsv"), "w") as fd:
        w = csv.writer(fd, delimiter="\t", lineterminator="\n")
        for row in out_manifest:
            w.writerow(row)
    with open(os.path.join(out_dir, "labels.tsv"), "w") as fd:
        w = csv.writer(fd, delimiter="\t", lineterminator="\n")
        for row in out_labels:
            w.writerow(row)

    # 同被试两个 trial 的电极集一样，随便从哪份 session 目录拷都行
    src_dir = list(feats.values())[0].dir
    shutil.copy(os.path.join(src_dir, "all_ordered_electrodes.json"), out_dir)
    for f in os.listdir(os.path.join(src_dir, "localization")):
        shutil.copy(os.path.join(src_dir, "localization", f),
                    os.path.join(out_dir, "localization", f))

    # ---- 写每个 fold 的 splits.json -----------------------------------------
    summary = {"eval_name": args.eval_name, "subject_id": sid, "trial_id": tid,
               "split": args.split, "window_samples": window,
               "n_neuroprobe_samples": n_total, "n_matched": n_matched,
               "coverage": coverage, "manifest": out_dir, "folds": []}

    for k, fold in enumerate(folds):
        splits = {}
        for name, key in (("train", "train_dataset"), ("val", "val_dataset"),
                          ("test", "test_dataset")):
            inner, idxs = unwrap(fold[key])
            splits[name] = [sample2row[(id(inner), i)] for i in idxs
                            if (id(inner), i) in sample2row]
        assert not (set(splits["train"]) & set(splits["test"])), "train/test 有重叠"
        assert not (set(splits["train"]) & set(splits["val"])), "train/val 有重叠"

        split_dir = os.path.join(repo, "saved_data_splits", "%s_fold%d" % (out_name, k))
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "splits.json"), "w") as fd:
            json.dump(splits, fd)
        print("fold %d: train=%d val=%d test=%d -> %s"
              % (k, len(splits["train"]), len(splits["val"]), len(splits["test"]), split_dir))
        summary["folds"].append({"fold": k, "split_dir": split_dir,
                                 "n_train": len(splits["train"]),
                                 "n_val": len(splits["val"]),
                                 "n_test": len(splits["test"])})

    with open(os.path.join(out_dir, "neuroprobe_summary.json"), "w") as fd:
        json.dump(summary, fd, indent=2)
    print("\nmanifest: %s" % out_dir)


if __name__ == "__main__":
    main()
