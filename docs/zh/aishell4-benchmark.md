# AISHELL-4 Benchmark

AISHELL-4 是中文会议场景数据集，适合验证 `openvad` 在远场、多说话人、重叠
语音、会议噪声下的效果。OpenSLR SLR111 提供完整数据，Hugging Face 上也有
镜像。官方数据说明显示：数据为 16 kHz、16-bit，约 120 小时；训练集约
107.50 小时，评测集约 12.72 小时。

## 准备依赖

```bash
uv sync --extra dev --extra bench
```

`bench` 额外安装：

- `huggingface_hub`：可选，用于从 Hugging Face 拉取 AISHELL-4。
- `soundfile`：用于读取 Hugging Face 镜像中的 FLAC/OGG 音频。

## 数据来源

OpenSLR：

```text
https://www.openslr.org/111/
```

Hugging Face：

```text
https://huggingface.co/datasets/AISHELL/AISHELL-4
```

完整 `test` split 约 5.2G，不建议在普通 CI 中自动下载。

## 推荐目录

```text
data/aishell4/
  test/
    wav/
    TextGrid/
```

如果你已经从 OpenSLR 下载并解压，只需要保证每个音频文件能和同名或后缀同名
的 `.rttm` 标注匹配。

## 从 Hugging Face 拉取 test split

```bash
uv run python tools/aishell4_prepare.py \
  --root data/aishell4 \
  --split test \
  --download-hf
```

该命令会下载 `test/wav/*` 和 `test/TextGrid/*.rttm`，并生成：

```text
data/aishell4/test_manifest.jsonl
```

## 只下载几个指定录音

先列出 Hugging Face 上 `test` split 的文件名：

```bash
uv run python tools/aishell4_prepare.py --split test --list-hf-files
```

然后选择几个录音 stem，只下载对应音频和 RTTM：

```bash
uv run python tools/aishell4_prepare.py \
  --root data/aishell4 \
  --split test \
  --download-hf \
  --hf-stems L_R003S01C02,L_R003S02C01 \
  --output data/aishell4/test_small_manifest.jsonl
```

`--hf-stems` 使用逗号分隔，不需要文件扩展名。脚本会匹配：

- `test/wav/{stem}.*`
- `test/wav/*{stem}.*`
- `test/TextGrid/{stem}.rttm`
- `test/TextGrid/*{stem}.rttm`

## 只做 smoke benchmark

```bash
uv run python tools/aishell4_prepare.py \
  --root data/aishell4 \
  --split test \
  --max-files 3 \
  --output data/aishell4/test_smoke_manifest.jsonl

uv run python tools/evaluate_dataset.py data/aishell4/test_smoke_manifest.jsonl
```

## 完整评估

```bash
uv run python tools/evaluate_dataset.py data/aishell4/test_manifest.jsonl
```

输出包括：

- 文件数
- 音频总时长
- 处理耗时
- real-time factor
- frame-level precision / recall / F1
- 误检秒数
- 漏检秒数

## 参数扫描

```bash
uv run python tools/sweep_thresholds.py data/aishell4/test_manifest.jsonl \
  --aggressiveness 0,1,2,3 \
  --onset 0.50,0.54,0.58,0.62,0.66 \
  --offset 0.34,0.38,0.42,0.46,0.50 \
  --top 20
```

## 可视化单条会议

```bash
uv run python tools/inspect_file.py data/aishell4/test/wav/example.flac \
  --output tests/reports/aishell4_example.html
```

## 注意事项

- AISHELL-4 是多说话人会议数据，RTTM 中可能存在重叠说话人。当前转换脚本会把
  所有说话人的 RTTM 区间取并集，作为“是否存在任意语音”的 VAD 标注。
- 如果目标是 diarization，不应该只看 VAD F1；但对 `openvad` 来说，任意说话人
  speech/non-speech 是合理 benchmark。
- 当前 `evaluate_dataset.py` 是帧级重叠判定，没有 collar。后续可以增加
  100-250 ms collar 来更贴近部分 SAD 评测协议。

## 资料来源

- OpenSLR SLR111: <https://www.openslr.org/111/>
- Hugging Face AISHELL-4: <https://huggingface.co/datasets/AISHELL/AISHELL-4>
- AISHELL-4 论文: <https://arxiv.org/abs/2104.03603>
- AISHELL-4 数据说明: <https://aishell-4.oss-cn-hangzhou.aliyuncs.com/AISHELL-4%20Data-Specification.pdf>
