"""把各 fold 的 results.json 汇总成一张表，并跟 leaderboard 数字对照。"""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--prefix", required=True, help="例如 np_volume_sub1_trial1")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(
        args.repo_dir, "outputs", args.prefix + "_fold*", "results.json")))
    if not paths:
        print("没找到 results.json，前缀: %s" % args.prefix)
        return

    aucs, f1s = [], []
    print("\n%-10s %10s %10s %8s" % ("fold", "roc_auc", "f1", "n_test"))
    for path in paths:
        with open(path) as fd:
            r = json.load(fd)
        fold = os.path.basename(os.path.dirname(path))
        aucs.append(r["roc_auc"])
        f1s.append(r["f1"])
        print("%-10s %10.4f %10.4f %8d"
              % (fold.replace(args.prefix + "_", ""), r["roc_auc"], r["f1"],
                 len(r.get("predicts", []))))

    mean_auc = sum(aucs) / len(aucs)
    print("%-10s %10.4f %10.4f" % ("mean", mean_auc, sum(f1s) / len(f1s)))

    eval_name = args.prefix.split("_")[1]
    ref = LEADERBOARD_POPT.get(eval_name)
    if ref is not None:
        print("\nleaderboard PopT (%s): %.3f    本次: %.3f    差值: %+.3f"
              % (eval_name, ref, mean_auc, mean_auc - ref))
        print("注意 leaderboard 的数字是多个 session 平均，这里只有一个 session。")


if __name__ == "__main__":
    main()
