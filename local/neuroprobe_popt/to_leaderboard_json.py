"""把 outputs/ 里散落的 results.json 汇总成 leaderboard 要的格式 + 一张对照表。

输出：
  local/neuroprobe_popt/leaderboard/<Split-Name>/population_<TASK>.json
      结构见 submit_neuroprobe/SUBMIT.md「Formatting results」一节
  外加一张 stdout 表格，跟 leaderboard 上 PopT 的数字逐任务对照。

注意 train_accuracy / train_roc_auc 填的是 -1 —— PopT 的 runner 只把测试集的
predicts/labels 写进 results.json（runner.py:200-208），没有训练集那一遍。
真要投稿得先给 runner 加一次训练集评估。
"""
import argparse
import glob
import json
import os

# https://neuroprobe.dev leaderboard, "Population Transformer (global z-scoring)",
# Geeling Chau / Caltech, 2026-04-28
LEADERBOARD_POPT = {
    "onset": 0.930, "speech": 0.919, "volume": 0.766, "delta_volume": 0.781,
    "pitch": 0.596, "word_index": 0.758, "word_gap": 0.620,
    "gpt2_surprisal": 0.603, "word_head_pos": 0.601, "word_part_speech": 0.604,
    "word_length": 0.622, "global_flow": 0.621, "local_flow": 0.618,
    "frame_brightness": 0.496, "face_num": 0.517,
}

SPLIT_DIR_NAME = {"within_session": "Within-Session",
                  "cross_session": "Cross-Session",
                  "cross_subject": "Cross-Subject"}


def metrics(res):
    """从 results.json 的 predicts(sigmoid 概率) / labels 算 accuracy 和 roc_auc。"""
    preds, labels = res.get("predicts"), res.get("labels")
    acc = None
    if preds and labels and len(preds) == len(labels):
        hit = sum(1 for p, y in zip(preds, labels) if (p > 0.5) == (y > 0.5))
        acc = hit / float(len(labels))
    return acc, res.get("roc_auc")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--split", default="within_session", choices=list(SPLIT_DIR_NAME))
    p.add_argument("--weights", default="popt_brainbert_stft")
    p.add_argument("--out-root", default=None)
    p.add_argument("--model-name", default="PopulationTransformer (reproduction)")
    p.add_argument("--author", default="")
    p.add_argument("--organization", default="")
    p.add_argument("--organization-url", default="")
    args = p.parse_args()

    out_root = args.out_root or os.path.join(
        args.repo_dir, "local", "neuroprobe_popt", "leaderboard")
    out_dir = os.path.join(out_root, SPLIT_DIR_NAME[args.split])

    # outputs/np_<split>_<task>_sub<S>_trial<T>_fold<k>_<weights>/results.json
    pattern = os.path.join(args.repo_dir, "outputs",
                           "np_%s_*_fold*_%s" % (args.split, args.weights),
                           "results.json")
    per_task = {}   # task -> {session_key -> {fold -> (acc, auc)}}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(os.path.dirname(path))
        body = name[len("np_%s_" % args.split):-len("_%s" % args.weights)]
        head, fold_part = body.rsplit("_fold", 1)
        task, sub_part, trial_part = head.rsplit("_", 2)
        session = "btbank%s_%s" % (sub_part[len("sub"):], trial_part[len("trial"):])
        with open(path) as fd:
            acc, auc = metrics(json.load(fd))
        per_task.setdefault(task, {}).setdefault(session, {})[int(fold_part)] = (acc, auc)

    if not per_task:
        print("没找到结果，找的是 %s" % pattern)
        return

    os.makedirs(out_dir, exist_ok=True)
    print("%-18s %8s %8s %8s %8s" % ("task", "sessions", "mean_auc", "leaderbd", "diff"))
    for task in sorted(per_task):
        sessions = per_task[task]
        results = {}
        all_aucs = []
        for session in sorted(sessions):
            folds = []
            for k in sorted(sessions[session]):
                acc, auc = sessions[session][k]
                all_aucs.append(auc)
                folds.append({"train_accuracy": -1.0, "train_roc_auc": -1.0,
                              "test_accuracy": acc, "test_roc_auc": auc})
            results[session] = {"population": {"one_second_after_onset": {
                "time_bin_start": 0.0, "time_bin_end": 1.0, "folds": folds}}}

        payload = {"model_name": args.model_name,
                   "description": "PopT (BrainBERT STFT) on Neuroprobe official splits",
                   "author": args.author, "organization": args.organization,
                   "organization_url": args.organization_url, "timestamp": 0,
                   "evaluation_results": results}
        with open(os.path.join(out_dir, "population_%s.json" % task), "w") as fd:
            json.dump(payload, fd, indent=2)

        mean_auc = sum(all_aucs) / len(all_aucs)
        ref = LEADERBOARD_POPT.get(task)
        print("%-18s %8d %8.3f %8s %8s"
              % (task, len(sessions), mean_auc,
                 "%.3f" % ref if ref else "-",
                 "%+.3f" % (mean_auc - ref) if ref else "-"))

    n_expected = 12 if args.split != "cross_subject" else 10
    print("\n写到 %s" % out_dir)
    print("提醒：leaderboard 要 15 任务 × %d session 全填满；" % n_expected)
    print("      train_accuracy / train_roc_auc 现在是 -1，投稿前要补训练集那一遍。")


if __name__ == "__main__":
    main()
