# 验证工具

`tools/` 目录包含一些小脚本，用于验证检测器在合成数据和带标注音频上的表现。

## 生成合成数据

```bash
uv run python tools/make_synthetic_dataset.py --output data/synth --count 20
```

该命令会生成 WAV 文件和 `manifest.jsonl` 标注。每行包含音频路径和参考语音
区间，单位为秒。

## 评估带标注音频

```bash
uv run python tools/evaluate_dataset.py data/synth/manifest.jsonl
```

评估器会输出帧级 precision、recall、F1、误检时长、漏检时长和处理速度。

Manifest 格式：

```json
{"audio": "sample_000.wav", "segments": [[0.3, 1.1], [1.7, 2.4]]}
```

相对音频路径会按 manifest 文件所在目录解析。

## 参数扫描

```bash
uv run python tools/sweep_thresholds.py data/synth/manifest.jsonl \
  --aggressiveness 0,1,2 \
  --onset 0.50,0.58,0.66 \
  --offset 0.34,0.42,0.50
```

在修改默认参数或为特定场景调参前，建议先运行参数扫描。

## 单文件可视化

```bash
uv run python tools/inspect_file.py speech.wav --output report.html
```

HTML 报告会展示检测出的语音段、帧级概率、能量和过零率。如果传入
`--labels labels.json`，还会同时展示参考标注区间。

标注文件格式：

```json
[[0.3, 1.1], [1.7, 2.4]]
```
