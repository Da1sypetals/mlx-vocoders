# mlx-vocoders

该 workspace 包含 Pupu-Vocoder Large 与 PC-NSF-HiFiGAN 的完整 Rust/MLX 推理实现。模型内部统一使用 MLX 原生 NLC 布局；PyTorch checkpoint 中的卷积权重在转换阶段完成布局变换和 weight norm 合并。

实现对应以下上游代码：

- [AliasingFreeNeuralAudioSynthesis](https://github.com/sizigi/AliasingFreeNeuralAudioSynthesis)
- [OpenVPI vocoders](https://github.com/openvpi/vocoders)
- PC-NSF-HiFiGAN 模型代码对应 DiffSinger commit `9b82b5d01680d6ae53e73e03c4de34d674d8e304`
- FCPE 使用 [fcpe-mlx](https://github.com/Da1sypetals/fcpe-mlx) submodule

## Checkpoint 转换

在 workspace 目录执行：

```bash
python tools/convert_checkpoints.py
python tools/verify_checkpoints.py
```

转换器生成：

- `../checkpoints/pupuvocoder_large/model.mlx.safetensors`
- `../checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.mlx.safetensors`

Pupu-Vocoder 使用 checkpoint 配置中的 `fmin=0`、`fmax=22050`。PC-NSF-HiFiGAN 使用 `fmin=40`、`fmax=16000`。

## 推理

Pupu-Vocoder：

```bash
cargo run --release -p pupu-vocoder --example infer -- \
  ../checkpoints/pupuvocoder_large/model.mlx.safetensors \
  '../audio/马文才-dry1_0m28s-1m12s.wav' \
  artifacts/pupu.wav
```

PC-NSF-HiFiGAN：

```bash
cargo run --release -p pc-nsf-hifigan --example infer -- \
  ../checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.mlx.safetensors \
  ../checkpoints/fcpe.safetensors \
  '../audio/马文才-dry1_0m28s-1m12s.wav' \
  artifacts/pc_nsf_hifigan.wav
```

两个 example 都执行 WAV 解码、单声道混合、44.1 kHz 重采样、mel 预处理、完整生成器推理和 PCM16 WAV 写出。PC example 额外执行 FCPE 推理及 F0 时间轴对齐。

## PyTorch/MLX 数值核对

`dump_features` examples 输出 mel、F0/source、conv-pre、每个 upsample stage 和最终波形。`tools/reference_inference.py` 使用上游 PyTorch 代码在 CPU 上输出同名 features，`tools/compare_features.py` 逐项报告 shape、最大绝对误差、平均绝对误差和最大相对误差。

```bash
cargo run --release -p pupu-vocoder --example dump_features -- \
  ../checkpoints/pupuvocoder_large/model.mlx.safetensors \
  '../audio/马文才-dry1_0m28s-1m12s.wav' \
  artifacts/trace/pupu_mlx

python tools/reference_inference.py pupu \
  '../audio/马文才-dry1_0m28s-1m12s.wav' \
  artifacts/trace/pupu_mlx \
  artifacts/trace/pupu_torch

python tools/compare_features.py \
  artifacts/trace/pupu_torch \
  artifacts/trace/pupu_mlx
```

PC 的 `dump_features` 参数顺序与 `infer` 一致，并在输出目录保存 `f0.npy` 供 PyTorch 模型使用完全相同的 F0 输入。

PC vocoder checkpoint 由 OpenVPI Team 以 CC BY-NC-SA 4.0 发布；使用和再分发权重时需遵守 checkpoint 目录中的 NOTICE。
