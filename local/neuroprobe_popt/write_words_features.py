"""给一个 session 的每个词都写一份 PopT/BrainBERT 特征。

为什么要这个：upstream 的 data.write_multi_subject_multi_channel 只给「当前任务
用到的那部分词」写特征 —— rms/pitch 只写上下四分位，pos 只写名词动词。Neuroprobe
有 15 个任务，每个任务的四分位落在不同的词上，逐任务提特征要重复 15 次，而且
build_neuroprobe_splits.py 按 start 时间对齐时会大面积对不上。

这里注册一个 task_name="all_words"：一次写完整个 trial 的所有词，后面 15 个任务
共用同一份特征，只换标签和 fold 索引。

零改动 upstream：import 进来打补丁，再调它自己的 hydra main。
用法见 local/neuroprobe_popt/1_write_words_features.sh
"""
import data.write_multi_subject_multi_channel as W
from data.subject_data import SubjectData

_orig_get_subject_data = W.get_subject_data
_orig_get_raw_data_and_labels = W.get_raw_data_and_labels


def _get_subject_data(data_cfg_template, task_name, index_subsample=None):
    if task_name == "all_words":
        return SubjectData(data_cfg_template, index_subsample=index_subsample)
    return _orig_get_subject_data(data_cfg_template, task_name,
                                  index_subsample=index_subsample)


def _get_raw_data_and_labels(subject_data, task_name):
    if task_name != "all_words":
        return _orig_get_raw_data_and_labels(subject_data, task_name)

    word_df = subject_data.words
    # SubjectData 在 index_subsample=None 时会 assert 这两个是对齐的，这里再确认一遍
    assert list(word_df.index) == list(range(len(word_df))), "词索引不连续"
    seeg_data = subject_data.neural_data
    assert seeg_data.shape[1] == len(word_df), \
        "神经数据样本数 %d != 词数 %d" % (seeg_data.shape[1], len(word_df))

    # labels.tsv 两列：第 0 列是词在 transcript 里的序号（占位，build_neuroprobe_splits.py
    # 会用真标签重写），第 1 列是词的 onset 时间，用来跟 Neuroprobe 的样本对齐。
    labels = list(zip(list(word_df.index), list(word_df["start"])))
    return seeg_data, labels


W.get_subject_data = _get_subject_data
W.get_raw_data_and_labels = _get_raw_data_and_labels

if __name__ == "__main__":
    W.main()
