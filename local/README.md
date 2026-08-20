# local/ —— bci_motor 项目专用运行脚本

这个目录是 fork 里新增的，**upstream 不存在**。目的是把「本机路径 + 实验配置」
和 upstream 代码彻底分开：

- upstream 的 `0_*.sh` ~ `5_*.sh` 和所有 `.py` 保持零改动，
  这样 `git fetch upstream && git rebase upstream/main` 基本不会冲突。
- 本机路径只写在 `env.sh` 一个文件里，而且它进了 `.gitignore`。

## 首次使用

```bash
cp local/env.sh.example local/env.sh
# 按本机情况改 env.sh（主要是 BrainTreebank 数据路径）
```

`REPO_DIR` 不用配置，`_common.sh` 会从脚本自身位置推导。

## 跑流程

脚本从任意目录执行都可以（`_common.sh` 会 `cd` 到仓库根）：

```bash
# 预训练（CPU 写数据 → GPU 训练）
./local/0_write_pretraining_data.sh
./local/1_create_pretraining_manifest.sh
./local/2_run_pretraining.sh          # 需要 GPU

# 微调（同一个 TASK 要贯穿 3 → 4 → 5）
TASK=pos ./local/3_write_finetuning_data.sh
TASK=pos ./local/4_create_finetuning_manifest.sh
TASK=pos ./local/5_run_finetuning.sh  # 需要 GPU
```

可覆盖的变量：`TASK`、`SUBJECT`、`N`、`NAME`、`WEIGHTS`。不传就用脚本里的默认值。

跑 GPU 的步骤记得先按 `bci_motor/CLAUDE.md` 里的流程 `salloc` 到计算节点，
别在登录节点上跑。长任务用 `local/slurm/` 下的 sbatch 模板。

## 两个坑

1. **`TASK` 必须贯穿 3 → 4 → 5**。`write_trial_data` 会跳过已存在的
   subject/trial 输出目录，如果两个 task 共用一个输出目录，第二个 task 会静默
   复用第一个 task 缓存的 npy，训出来的结果是错的但不报错。

2. **从 HuggingFace 下权重时一定要 `local_dir=pretrained_weights/`**。
   `snapshot_download` 默认会把整个 HF repo（含 `README.md`、`.gitattributes`）
   铺到给定目录，铺到仓库根就会覆盖 upstream 的 README。
