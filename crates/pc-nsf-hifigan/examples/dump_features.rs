use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use mlx_rs::ops::indexing::IndexOp;
use pc_nsf_hifigan::{FcpeEstimator, HOP_SIZE, PcNsfHifigan, load_mono_audio};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let checkpoint =
        PathBuf::from(args.next().context(
            "用法: dump_features <checkpoint> <fcpe> <input.wav> <output_dir> [samples]",
        )?);
    let fcpe_checkpoint =
        PathBuf::from(args.next().context(
            "用法: dump_features <checkpoint> <fcpe> <input.wav> <output_dir> [samples]",
        )?);
    let input =
        PathBuf::from(args.next().context(
            "用法: dump_features <checkpoint> <fcpe> <input.wav> <output_dir> [samples]",
        )?);
    let output_dir =
        PathBuf::from(args.next().context(
            "用法: dump_features <checkpoint> <fcpe> <input.wav> <output_dir> [samples]",
        )?);
    let samples = args
        .next()
        .map(|value| value.to_string_lossy().parse::<i32>())
        .transpose()?;

    fs::create_dir_all(&output_dir)?;
    let (audio, _) = load_mono_audio(input)?;
    let audio = match samples {
        Some(length) => audio.index((.., ..length)),
        None => audio,
    };
    let mut model = PcNsfHifigan::load(checkpoint)?;
    let mut estimator = FcpeEstimator::load(fcpe_checkpoint);
    let f0 = estimator.estimate(&audio, audio.shape()[1] / HOP_SIZE as i32)?;
    let features = model.infer_with_features(&audio, &f0)?;
    audio.save_numpy(output_dir.join("audio.npy"))?;
    features.mel.save_numpy(output_dir.join("mel.npy"))?;
    features.f0.save_numpy(output_dir.join("f0.npy"))?;
    features.source.save_numpy(output_dir.join("source.npy"))?;
    features
        .conv_pre
        .save_numpy(output_dir.join("conv_pre.npy"))?;
    for (index, resblock) in features.resblocks.iter().enumerate() {
        resblock.save_numpy(output_dir.join(format!(
            "stage_{}_resblock_{}.npy",
            index / 3,
            index % 3
        )))?;
    }
    for (index, stage) in features.stages.iter().enumerate() {
        stage.save_numpy(output_dir.join(format!("stage_{index}.npy")))?;
    }
    features.output.save_numpy(output_dir.join("output.npy"))?;
    Ok(())
}
