use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv1d, Conv1dBuilder, ConvTranspose1d, ConvTranspose1dBuilder, leaky_relu};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{PadMode, pad, tanh};

use crate::MelSpectrogram;

const UPSAMPLE_RATES: [i32; 5] = [8, 8, 2, 2, 2];
const UPSAMPLE_KERNELS: [i32; 5] = [16, 16, 4, 4, 4];
const RESBLOCK_KERNELS: [i32; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [i32; 3] = [1, 3, 5];
const INITIAL_CHANNELS: i32 = 512;
const SOURCE_SAMPLE_RATE: f32 = 5_512.5;
const SOURCE_UPSAMPLE: i32 = 64;

fn weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .with_context(|| format!("缺少权重 {key}"))
        .cloned()
}

fn conv1d_layer(
    weights: &HashMap<String, Array>,
    prefix: &str,
    input_channels: i32,
    output_channels: i32,
    kernel_size: i32,
    padding: i32,
    dilation: i32,
) -> Result<Conv1d> {
    let mut layer = Conv1dBuilder::new(input_channels, output_channels, kernel_size)
        .padding(padding)
        .dilation(dilation)
        .build()?;
    layer.weight.value = weight(weights, &format!("{prefix}.weight"))?;
    layer.bias.value = weights.get(&format!("{prefix}.bias")).cloned();
    Ok(layer)
}

fn conv_transpose1d_layer(
    weights: &HashMap<String, Array>,
    prefix: &str,
    input_channels: i32,
    output_channels: i32,
    kernel_size: i32,
    stride: i32,
    padding: i32,
) -> Result<ConvTranspose1d> {
    let mut layer = ConvTranspose1dBuilder::new(input_channels, output_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .build()?;
    layer.weight.value = weight(weights, &format!("{prefix}.weight"))?;
    layer.bias.value = weights.get(&format!("{prefix}.bias")).cloned();
    Ok(layer)
}

struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn load(
        weights: &HashMap<String, Array>,
        index: usize,
        channels: i32,
        kernel_size: i32,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        for (layer, dilation) in RESBLOCK_DILATIONS.into_iter().enumerate() {
            convs1.push(conv1d_layer(
                weights,
                &format!("resblocks.{index}.convs1.{layer}"),
                channels,
                channels,
                kernel_size,
                (kernel_size * dilation - dilation) / 2,
                dilation,
            )?);
            convs2.push(conv1d_layer(
                weights,
                &format!("resblocks.{index}.convs2.{layer}"),
                channels,
                channels,
                kernel_size,
                (kernel_size - 1) / 2,
                1,
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&mut self, input: &Array) -> Result<Array> {
        let mut output = input.clone();
        for layer in 0..3 {
            let hidden = self.convs1[layer].forward(&leaky_relu(&output, 0.1)?)?;
            let hidden = self.convs2[layer].forward(&leaky_relu(&hidden, 0.1)?)?;
            output = output.add(&hidden)?;
        }
        Ok(output)
    }
}

pub struct PcNsfFeatures {
    pub mel: Array,
    pub f0: Array,
    pub source: Array,
    pub conv_pre: Array,
    pub resblocks: Vec<Array>,
    pub stages: Vec<Array>,
    pub output: Array,
}

pub struct PcNsfHifigan {
    mel: MelSpectrogram,
    conv_pre: Conv1d,
    upsamplers: Vec<ConvTranspose1d>,
    source_conv: Conv1d,
    resblocks: Vec<ResBlock1>,
    conv_post: Conv1d,
}

impl PcNsfHifigan {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let weights = Array::load_safetensors(path)?;
        let mel = MelSpectrogram::from_weights(&weights)?;
        let conv_pre = conv1d_layer(&weights, "conv_pre", 128, INITIAL_CHANNELS, 7, 3, 1)?;

        let mut upsamplers = Vec::with_capacity(5);
        let mut resblocks = Vec::with_capacity(15);
        let mut channels = INITIAL_CHANNELS;
        for stage in 0..5 {
            let output_channels = channels / 2;
            upsamplers.push(conv_transpose1d_layer(
                &weights,
                &format!("ups.{stage}"),
                channels,
                output_channels,
                UPSAMPLE_KERNELS[stage],
                UPSAMPLE_RATES[stage],
                (UPSAMPLE_KERNELS[stage] - UPSAMPLE_RATES[stage]) / 2,
            )?);
            for (kernel_index, kernel_size) in RESBLOCK_KERNELS.into_iter().enumerate() {
                resblocks.push(ResBlock1::load(
                    &weights,
                    stage * 3 + kernel_index,
                    output_channels,
                    kernel_size,
                )?);
            }
            channels = output_channels;
        }

        Ok(Self {
            mel,
            conv_pre,
            upsamplers,
            source_conv: conv1d_layer(&weights, "source_conv", 1, 128, 1, 0, 1)?,
            resblocks,
            conv_post: conv1d_layer(&weights, "conv_post", channels, 1, 7, 3, 1)?,
        })
    }

    pub fn infer(&mut self, audio: &Array, f0: &Array) -> Result<Array> {
        Ok(self.infer_internal(audio, f0, false)?.output)
    }

    pub fn infer_with_features(&mut self, audio: &Array, f0: &Array) -> Result<PcNsfFeatures> {
        self.infer_internal(audio, f0, true)
    }

    fn infer_internal(
        &mut self,
        audio: &Array,
        f0: &Array,
        capture_features: bool,
    ) -> Result<PcNsfFeatures> {
        let mel = self.mel.forward(audio)?;
        let frames = mel.shape()[1];
        anyhow::ensure!(
            f0.shape() == [mel.shape()[0], frames],
            "F0 shape {:?} 与 mel shape {:?} 不匹配",
            f0.shape(),
            mel.shape()
        );

        let n = Array::arange::<_, f32>(Some(1), SOURCE_UPSAMPLE + 1, None)?.reshape(&[
            1,
            1,
            SOURCE_UPSAMPLE,
        ])?;
        let s0 = f0
            .reshape(&[f0.shape()[0], frames, 1])?
            .divide(Array::from(SOURCE_SAMPLE_RATE))?;
        let difference = s0
            .index((.., 1.., ..))
            .subtract(s0.index((.., ..-1i32, ..)))?;
        let ds0 = pad(
            &difference,
            &[(0, 0), (0, 1), (0, 0)],
            Array::from(0.0f32),
            Some(PadMode::Constant),
        )?;
        let rad = s0.multiply(&n)?.add(
            &ds0.multiply(Array::from(0.5f32))?
                .multiply(&n)?
                .multiply(&n.subtract(Array::from(1.0f32))?)?
                .divide(Array::from(SOURCE_UPSAMPLE as f32))?,
        )?;
        let rad2 = rad
            .index((.., .., -1i32..))
            .add(Array::from(0.5f32))?
            .remainder(Array::from(1.0f32))?
            .subtract(Array::from(0.5f32))?;
        let accumulated = rad2.cumsum(1, None, None)?.remainder(Array::from(1.0f32))?;
        let accumulated = pad(
            accumulated.index((.., ..-1i32, ..)),
            &[(0, 0), (1, 0), (0, 0)],
            Array::from(0.0f32),
            Some(PadMode::Constant),
        )?;
        let source = rad
            .add(&accumulated)?
            .reshape(&[f0.shape()[0], frames * SOURCE_UPSAMPLE, 1])?
            .multiply(Array::from(2.0f32 * std::f32::consts::PI))?
            .sin()?;

        let mut hidden = self.conv_pre.forward(&mel)?;
        let conv_pre = hidden.clone();
        let mut resblocks = Vec::with_capacity(if capture_features { 15 } else { 0 });
        let mut stages = Vec::with_capacity(if capture_features { 5 } else { 0 });
        for stage in 0..5 {
            hidden = self.upsamplers[stage].forward(&leaky_relu(&hidden, 0.1)?)?;
            if stage == 1 {
                hidden = hidden.add(&self.source_conv.forward(&source)?)?;
            }
            let first = self.resblocks[stage * 3].forward(&hidden)?;
            let second = self.resblocks[stage * 3 + 1].forward(&hidden)?;
            let third = self.resblocks[stage * 3 + 2].forward(&hidden)?;
            if capture_features {
                resblocks.extend([first.clone(), second.clone(), third.clone()]);
            }
            hidden = first
                .add(&second)?
                .add(&third)?
                .divide(Array::from(3.0f32))?;
            hidden.eval()?;
            if capture_features {
                stages.push(hidden.clone());
            }
        }

        let output = tanh(&self.conv_post.forward(&leaky_relu(&hidden, None)?)?)?;
        output.eval()?;
        Ok(PcNsfFeatures {
            mel,
            f0: f0.clone(),
            source,
            conv_pre,
            resblocks,
            stages,
            output,
        })
    }
}
