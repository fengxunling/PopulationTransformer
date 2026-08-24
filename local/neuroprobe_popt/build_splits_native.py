"""从 neuroprobe 原生特征缓存组装 PopT 能吃的 manifest + splits.json。

配套 write_neuroprobe_features.py。因为窗口是直接用 neuroprobe 的 est_idx 定位的，
不需要按 onset 时间对齐，覆盖率天然 100%，onset/speech 也能做。
"""
import argparse
import csv
import json
import os
import shutil
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--braintreebank-root", required=True)
    p.add_argument("--feat-dir", required=True,
                   help="write_neuroprobe_features.py 的输出目录（含 feats/）")
    p.add_argument("--subject-id", type=int, default=1)
    p.add_argument("--trial-id", type=int, default=1)
    p.add_argument("--eval-name", required=True)
    p.add_argument("--split", default="within_session",
                   choices=["within_session", "cross_session"])
    p.add_argument("--out-name", required=True)
    return p.parse_args()


def unwrap(ds):
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

    import torch
    import neuroprobe.config as npcfg
    from neuroprobe import BrainTreebankSubject
    import neuroprobe.train_test_splits as tts

    sid, tid = args.subject_id, args.trial_id
    subject_name = "sub_%d" % sid

    meta = json.load(open(os.path.join(args.feat_dir, "meta.json")))
    before, after = meta["before_samples"], meta["after_samples"]

    subject = BrainTreebankSubject(sid, allow_corrupted=False, cache=False,
                                   dtype=torch.float32)
    kw = dict(dtype=torch.float32, output_indices=True, output_dict=False,
              start_neural_data_before_word_onset=before,
              end_neural_data_after_word_onset=after, lite=True)
    if args.split == "within_session":
        folds = tts.generate_splits_within_session(subject, tid, args.eval_name, **kw)
    else:
        folds = tts.generate_splits_cross_session(subject, tid, args.eval_name, **kw)

    # 一个 (dataset, 样本索引) 对应一个全局行号
    out_manifest, out_labels = [], []
    sample2row = {}
    seen_inner = {}
    missing = 0
    for fold in folds:
        for key in ("train_dataset", "val_dataset", "test_dataset"):
            inner, _ = unwrap(fold[key])
            if id(inner) in seen_inner:
                continue
            seen_inner[id(inner)] = inner
            feats = os.path.join(args.feat_dir, "feats")   # 同 subject 的两个 trial 共用？不，按 trial 分目录
            for i in range(inner.n_samples):
                (wf, _), label = inner[i]
                path = os.path.join(feats, "%d.npy" % int(wf))
                if not os.path.exists(path):
                    missing += 1
                    continue
                sample2row[(id(inner), i)] = len(out_manifest)
                out_manifest.append((path, subject_name))
                out_labels.append((str(bool(label)), int(wf)))

    total = sum(inner.n_samples for inner in seen_inner.values())
    print("样本 %d，特征缺失 %d，覆盖 %.1f%%"
          % (total, missing, 100.0 * (total - missing) / max(total, 1)))
    if missing:
        sys.exit("有特征缺失，先把 write_neuroprobe_features.py 跑全")

    out_dir = os.path.join(repo, "saved_examples", args.out_name)
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
    shutil.copy(os.path.join(args.feat_dir, "all_ordered_electrodes.json"), out_dir)
    for f in os.listdir(os.path.join(args.feat_dir, "localization")):
        shutil.copy(os.path.join(args.feat_dir, "localization", f),
                    os.path.join(out_dir, "localization", f))

    for k, fold in enumerate(folds):
        splits = {}
        for name, key in (("train", "train_dataset"), ("val", "val_dataset"),
                          ("test", "test_dataset")):
            inner, idxs = unwrap(fold[key])
            splits[name] = [sample2row[(id(inner), i)] for i in idxs
                            if (id(inner), i) in sample2row]
        assert not (set(splits["train"]) & set(splits["test"])), "train/test 重叠"
        assert not (set(splits["train"]) & set(splits["val"])), "train/val 重叠"
        split_dir = os.path.join(repo, "saved_data_splits", "%s_fold%d" % (args.out_name, k))
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "splits.json"), "w") as fd:
            json.dump(splits, fd)
        print("fold %d: train=%d val=%d test=%d"
              % (k, len(splits["train"]), len(splits["val"]), len(splits["test"])))

    print("manifest: %s" % out_dir)


if __name__ == "__main__":
    main()
