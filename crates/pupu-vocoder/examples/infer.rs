use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};
use pupu_vocoder::{PupuVocoder, load_mono_audio, save_pcm16_wav};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = PathBuf::from(
        args.next()
            .context("用法: infer <model.mlx.safetensors> <input.wav> <output.wav>")?,
    );
    let input = PathBuf::from(
        args.next()
            .context("用法: infer <model.mlx.safetensors> <input.wav> <output.wav>")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("用法: infer <model.mlx.safetensors> <input.wav> <output.wav>")?,
    );

    let (audio, target_length) = load_mono_audio(input)?;
    let mut model = PupuVocoder::load(checkpoint)?;
    let waveform = model.infer(&audio, target_length)?;
    save_pcm16_wav(output, &waveform)?;
    Ok(())
}
