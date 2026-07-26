use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};
use pc_nsf_hifigan::{FcpeEstimator, HOP_SIZE, PcNsfHifigan, load_mono_audio, save_pcm16_wav};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = PathBuf::from(args.next().context(
        "用法: infer <model.mlx.safetensors> <fcpe.safetensors> <input.wav> <output.wav>",
    )?);
    let fcpe_checkpoint = PathBuf::from(args.next().context(
        "用法: infer <model.mlx.safetensors> <fcpe.safetensors> <input.wav> <output.wav>",
    )?);
    let input = PathBuf::from(args.next().context(
        "用法: infer <model.mlx.safetensors> <fcpe.safetensors> <input.wav> <output.wav>",
    )?);
    let output = PathBuf::from(args.next().context(
        "用法: infer <model.mlx.safetensors> <fcpe.safetensors> <input.wav> <output.wav>",
    )?);

    let (audio, _) = load_mono_audio(input)?;
    let mut model = PcNsfHifigan::load(checkpoint)?;
    let mut estimator = FcpeEstimator::load(fcpe_checkpoint);
    let f0 = estimator.estimate(&audio, audio.shape()[1] / HOP_SIZE as i32)?;
    let waveform = model.infer(&audio, &f0)?;
    save_pcm16_wav(output, &waveform)?;
    Ok(())
}
