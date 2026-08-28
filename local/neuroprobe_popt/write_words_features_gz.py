"""全局 z-score 版的特征提取（对应论文里的 "global z-scoring"）。

和 write_words_features.py 的唯一区别：STFT 之后的归一化统计量从「每个窗口内
按时间轴算」换成「整个 session 算一份」。论文 Supplementary Table 3 里这两种是
分开列的两行，差距很大：
    PopulationTransformer (off-the-shelf; per-window STFT z-scoring)  Overall 0.549
    Population Transformer (global z-scoring)                         Overall 0.670
BrainBERT 上同样的对比是 0.585 -> 0.626。

实现上不改任何 upstream 文件：
  - "all_words" 那两个补丁直接复用 write_words_features（import 时就打好了）
  - 归一化靠 MultiElecSpecPretrained.forward(wav, spec_preprocessed=...) 这个
    现成的钩子 —— 自己算好频谱传进去，就绕开了 stft.py 里写死的 per-window zscore

保留 upstream 的 [:, :, 10:-10] 裁剪。不保留的话帧数会变，BrainBERT 写死的
「取中间 10 帧」就会落到不同的时间段上，等于一次改了两个变量。
"""
import os
import numpy as np
from scipy import signal
from pathlib import Path
from tqdm import tqdm as tqdm

# import 即打上 all_words 的补丁
import local.neuroprobe_popt.write_words_features as _base
import data.write_multi_subject_multi_channel as W

SAMP_FREQ = 2048
# Pass A 的抽样步长：只是估 [n_elec, n_freq] 这几千个统计量，不用全量
STATS_STRIDE = int(os.environ.get("GZ_STATS_STRIDE", "5"))


def _abs_stft(x, cfg):
    """复现 preprocessors/stft.py 的 STFT 部分，到 np.abs 为止。

    x: [n_electrodes, n_time] -> [n_electrodes, n_freq, n_frames]
    """
    _, _, Zxx = signal.stft(x, SAMP_FREQ, nperseg=cfg.nperseg,
                            noverlap=cfg.noverlap, return_onesided=True)
    Zxx = Zxx[:, :cfg.freq_channel_cutoff]
    return np.abs(Zxx)


def _global_stats(seeg_data, cfg):
    """Pass A：算每个 (电极, 频道) 在整个 session 上的 mean / std。

    seeg_data: [n_elec, n_windows, n_time]
    返回 mean, std，形状都是 [n_elec, n_freq]
    """
    n_win = seeg_data.shape[1]
    idxs = range(0, n_win, STATS_STRIDE)
    total = cnt = None
    sq = None
    for i in tqdm(list(idxs), desc="pass A: 全局统计量"):
        a = _abs_stft(seeg_data[:, i], cfg)          # [n_elec, n_freq, n_frames]
        s = a.sum(axis=-1)
        q = (a ** 2).sum(axis=-1)
        n = a.shape[-1]
        if total is None:
            total, sq, cnt = s, q, n
        else:
            total += s; sq += q; cnt += n
    mean = total / cnt
    var = sq / cnt - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    std[std == 0] = 1.0        # 和 upstream zscore() 里同样的兜底
    return mean, std


def _write_outputs_gz(subject, trial, seeg_data, labels, ordered_electrodes,
                      extracter, output_path):
    cfg = extracter.spec_preprocessor.cfg
    out_dir = os.path.join(output_path, subject, trial)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    mean, std = _global_stats(seeg_data, cfg)
    stats_path = os.path.join(output_path, subject, "%s_global_stats.npz" % trial)
    np.savez(stats_path, mean=mean, std=std, stride=STATS_STRIDE,
             nperseg=cfg.nperseg, noverlap=cfg.noverlap,
             freq_channel_cutoff=cfg.freq_channel_cutoff)
    print("全局统计量 -> %s  (mean %s, std 中位数 %.4g)"
          % (stats_path, mean.shape, float(np.median(std))))

    mean_ = mean[:, :, None]
    std_ = std[:, :, None]
    manifest = []
    for idx in tqdm(range(len(labels)), desc="pass B: 提特征"):
        a = _abs_stft(seeg_data[:, idx], cfg)        # [n_elec, n_freq, n_frames]
        z = (a - mean_) / std_
        z = z[:, :, 10:-10]                          # 和 upstream 一致，保住帧数
        spec = np.transpose(z, [0, 2, 1])            # [n_elec, n_frames, n_freq]
        emb = extracter(None, spec_preprocessed=spec).numpy()
        save_path = os.path.join(out_dir, "%d.npy" % idx)
        np.save(save_path, emb)
        manifest.append(save_path)
    return manifest


W.write_outputs = _write_outputs_gz

if __name__ == "__main__":
    W.main()
