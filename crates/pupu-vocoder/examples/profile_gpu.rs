use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use metal::{CaptureDescriptor, CaptureManager, Device, MTLCaptureDestination};
use mlx_rs::Array;
use pupu_vocoder::{MelSpectrogram, PupuVocoder, load_mono_audio, save_pcm16_wav};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = PathBuf::from(args.next().context(
        "用法: profile_gpu <model.mlx.safetensors> <input.wav> <trace.gputrace> <output.wav>",
    )?);
    let input = PathBuf::from(args.next().context(
        "用法: profile_gpu <model.mlx.safetensors> <input.wav> <trace.gputrace> <output.wav>",
    )?);
    let trace = PathBuf::from(args.next().context(
        "用法: profile_gpu <model.mlx.safetensors> <input.wav> <trace.gputrace> <output.wav>",
    )?);
    let output = PathBuf::from(args.next().context(
        "用法: profile_gpu <model.mlx.safetensors> <input.wav> <trace.gputrace> <output.wav>",
    )?);
    let trace_name = trace
        .file_name()
        .context("GPU trace 路径缺少文件名")?
        .to_owned();
    let trace = trace
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .context("GPU trace 输出目录不存在")?
        .join(trace_name);

    let (audio, target_length) = load_mono_audio(input)?;
    let weights = Array::load_safetensors(&checkpoint)?;
    let mel = MelSpectrogram::from_weights(&weights)?.forward(&audio)?;
    mel.eval()?;
    let mut model = PupuVocoder::load(checkpoint)?;
    model.infer_mel(&mel, target_length)?.eval()?;

    let device = Device::system_default().context("找不到 Metal device")?;
    let manager = CaptureManager::shared();
    let destination = MTLCaptureDestination::GpuTraceDocument;
    if !manager.supports_destination(destination) {
        return Err(anyhow!("当前 Metal 工具链不支持导出 GPU trace"));
    }
    let descriptor = CaptureDescriptor::new();
    descriptor.set_capture_device(&device);
    descriptor.set_destination(destination);
    descriptor.set_output_url(trace);
    manager
        .start_capture(&descriptor)
        .map_err(|error| anyhow!(error))?;

    let waveform = model.infer_mel(&mel, target_length)?;
    waveform.eval()?;
    manager.stop_capture();

    save_pcm16_wav(output, &waveform)?;
    Ok(())
}
