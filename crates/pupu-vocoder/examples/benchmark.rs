use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use mlx_rs::Array;
use pupu_vocoder::{MelSpectrogram, PupuVocoder, load_mono_audio};

const WARMUP_ITERATIONS: usize = 3;

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint = PathBuf::from(
        args.next()
            .context("用法: benchmark <model.mlx.safetensors> <input.wav> <iterations>")?,
    );
    let input = PathBuf::from(
        args.next()
            .context("用法: benchmark <model.mlx.safetensors> <input.wav> <iterations>")?,
    );
    let iterations = args
        .next()
        .context("用法: benchmark <model.mlx.safetensors> <input.wav> <iterations>")?
        .to_string_lossy()
        .parse::<usize>()?;
    ensure!(iterations > 0, "iterations 必须大于 0");
    ensure!(args.next().is_none(), "参数数量错误");

    let (audio, target_length) = load_mono_audio(input)?;
    let weights = Array::load_safetensors(&checkpoint)?;
    let mel = MelSpectrogram::from_weights(&weights)?.forward(&audio)?;
    mel.eval()?;
    let mut model = PupuVocoder::load(checkpoint)?;

    let mut reference = Vec::new();
    for warmup in 0..WARMUP_ITERATIONS {
        let waveform = model.infer_mel(&mel, target_length)?;
        waveform.eval()?;
        if warmup + 1 == WARMUP_ITERATIONS {
            reference.extend_from_slice(waveform.as_slice::<f32>());
        }
    }

    let mut durations = Vec::with_capacity(iterations);
    let mut final_waveform = None;
    for iteration in 0..iterations {
        let start = Instant::now();
        let waveform = model.infer_mel(&mel, target_length)?;
        waveform.eval()?;
        let elapsed = start.elapsed();
        println!(
            "iteration {}: {:.3} ms",
            iteration + 1,
            elapsed.as_secs_f64() * 1e3
        );
        durations.push(elapsed);
        final_waveform = Some(waveform);
    }

    let final_waveform = final_waveform.expect("至少执行一次 benchmark");
    let output = final_waveform.as_slice::<f32>();
    ensure!(output.len() == reference.len(), "输出长度发生变化");
    let max_abs_diff = output
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);

    durations.sort_unstable();
    let total = durations.iter().copied().sum::<Duration>();
    let median = if iterations % 2 == 0 {
        (durations[iterations / 2 - 1] + durations[iterations / 2]) / 2
    } else {
        durations[iterations / 2]
    };
    println!("min: {:.3} ms", durations[0].as_secs_f64() * 1e3);
    println!("median: {:.3} ms", median.as_secs_f64() * 1e3);
    println!(
        "mean: {:.3} ms",
        total.as_secs_f64() * 1e3 / iterations as f64
    );
    println!("max abs diff: {max_abs_diff:.9}");
    Ok(())
}
