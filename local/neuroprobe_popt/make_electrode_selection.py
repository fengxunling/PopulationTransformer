"""电极选择 = PopT 的 clean_laplacian ∩ Neuroprobe lite。

为什么要取交集：
- Neuroprobe 规定用 NEUROPROBE_LITE_ELECTRODES（每个被试约 120 个），
  不用这份电极就没法跟 leaderboard 比。
- PopT 的预处理是 laplacian re-reference（conf/data/subject_data_template.yaml），
  data/h5_data_reader.py:52 要求电极在同一根电极杆上前后各有一个邻居，
  否则 assert len(nbrs)==2 直接挂。electrode_selections/clean_laplacian.json
  就是筛过的那一批。
交集大小：sub_1 88 / sub_2 89 / sub_3 82 / sub_4 101 / sub_7 104 / sub_10 95。
"""
import argparse
import json
import os

NEUROPROBE_SUBJECTS = [1, 2, 3, 4, 7, 10]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--braintreebank-root", default=".",
                   help="只为了满足 neuroprobe.config 的 import 检查，不读数据")
    args = p.parse_args()

    # neuroprobe.config 在 import 时就要求这个环境变量，但这里只读电极表，用不到数据
    os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", args.braintreebank_root)
    import neuroprobe.config as npcfg

    clean = json.load(open(os.path.join(
        args.repo_dir, "electrode_selections", "clean_laplacian.json")))

    out = {}
    for sid in NEUROPROBE_SUBJECTS:
        name = "sub_%d" % sid
        lite = npcfg.NEUROPROBE_LITE_ELECTRODES["btbank%d" % sid]
        keep = [e for e in lite if e in set(clean[name])]  # 保持 lite 的顺序
        out[name] = keep
        print("%-7s clean=%-4d lite=%-4d 交集=%d" % (name, len(clean[name]), len(lite), len(keep)))

    with open(args.out, "w") as fd:
        json.dump(out, fd, indent=1)
    print("\n写到 %s" % args.out)


if __name__ == "__main__":
    main()
