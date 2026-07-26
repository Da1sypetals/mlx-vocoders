use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use mlx_rs::Array;
use mlx_rs::ops::clip;
use soxr::{Soxr, format::Mono};

use crate::SAMPLE_RATE;

pub fn load_mono_audio(path: impl AsRef<Path>) -> Result<(Array, usize)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = i32::from(spec.channels);
    ensure!(channels > 0, "WAV 文件没有音频通道");

    let samples = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => {
            let scale = 2.0f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    ensure!(
        samples.len() % channels as usize == 0,
        "WAV 样本数无法按通道数整除"
    );

    let frames = samples.len() as i32 / channels;
    let interleaved = Array::from_slice(&samples, &[frames, channels]);
    let mono = interleaved.mean_axis(1, false)?.reshape(&[1, frames])?;
    mono.eval()?;
    let original_len = frames as usize;

    if spec.sample_rate == SAMPLE_RATE {
        return Ok((mono, original_len));
    }

    let input = mono.as_slice::<f32>();
    let target_len =
        (input.len() as f64 * f64::from(SAMPLE_RATE) / f64::from(spec.sample_rate)).ceil() as usize;
    let mut output = vec![0.0f32; target_len + 512];
    let mut resampler = Soxr::<Mono<f32>>::new(f64::from(spec.sample_rate), f64::from(SAMPLE_RATE))
        .map_err(|error| anyhow!("libsoxr 初始化失败: {error}"))?;
    let processed = resampler
        .process(input, &mut output)
        .map_err(|error| anyhow!("libsoxr 处理失败: {error}"))?;
    ensure!(
        processed.input_frames == input.len(),
        "libsoxr 未消费全部输入样本"
    );
    let drained = resampler
        .drain(&mut output[processed.output_frames..])
        .map_err(|error| anyhow!("libsoxr 排空失败: {error}"))?;
    let produced = processed.output_frames + drained;
    ensure!(
        produced == target_len,
        "libsoxr 输出长度 {produced}，期望 {target_len}"
    );
    output.truncate(produced);

    Ok((Array::from_slice(&output, &[1, produced as i32]), produced))
}

pub fn save_pcm16_wav(path: impl AsRef<Path>, audio: &Array) -> Result<()> {
    let flat = audio.reshape(&[-1])?;
    let scaled = clip(&flat, (-1.0f32, 1.0f32))?
        .multiply(Array::from(32_768.0f32))?
        .round(None)?;
    let pcm = clip(&scaled, (-32_768.0f32, 32_767.0f32))?.as_type::<i16>()?;
    pcm.eval()?;
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in pcm.as_slice::<i16>() {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}
