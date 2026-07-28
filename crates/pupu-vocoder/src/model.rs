use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv1d, Conv1dBuilder};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    PadMode, broadcast_to, concatenate_axis, conv_transpose1d, conv1d, eq, maximum, minimum, pad,
    r#where,
};
use mlx_rs::{Array, array};

use crate::MelSpectrogram;

const UPSAMPLE_RATES: [i32; 5] = [8, 8, 2, 2, 2];
const TOTAL_UPSAMPLE_RATES: [i32; 5] = [8, 64, 128, 256, 512];
const RESBLOCK_KERNELS: [i32; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [i32; 3] = [1, 3, 5];
const INITIAL_CHANNELS: i32 = 1_536;

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

struct Activation1d {
    alpha: Array,
    beta: Array,
    up_filter: Array,
    down_filter: Array,
    channels: i32,
}

impl Activation1d {
    fn load(weights: &HashMap<String, Array>, prefix: &str, channels: i32) -> Result<Self> {
        Ok(Self {
            alpha: weight(weights, &format!("{prefix}.act.alpha"))?,
            beta: weight(weights, &format!("{prefix}.act.beta"))?,
            up_filter: weight(weights, &format!("{prefix}.upsample.filter"))?,
            down_filter: weight(weights, &format!("{prefix}.downsample.lowpass.filter"))?,
            channels,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array> {
        let padded = pad(
            input,
            &[(0, 0), (5, 5), (0, 0)],
            Array::from(0.0f32),
            Some(PadMode::Edge),
        )?;
        let up_filter = broadcast_to(&self.up_filter, &[self.channels, 12, 1])?;
        let upsampled = conv_transpose1d(&padded, &up_filter, 2, 0, 1, 0, self.channels)?
            .multiply(array!(2.0f32))?;
        let upsampled_length = upsampled.shape()[1];
        let upsampled = upsampled.index((.., 15..upsampled_length - 15, ..));

        let alpha = self.alpha.exp()?;
        let beta = self.beta.exp()?;
        let batch = upsampled.shape()[0];
        let first = Array::zeros::<f32>(&[batch, 1, self.channels])?;
        let shifted = concatenate_axis(&[&first, &upsampled.index((.., ..-1i32, ..))], 1)?;
        let delta = upsampled.subtract(&shifted)?;
        let sum = upsampled.add(&shifted)?;
        let sinc_argument = alpha
            .multiply(&delta)?
            .divide(Array::from(std::f32::consts::PI))?;
        let sinc_phase = sinc_argument.multiply(Array::from(std::f32::consts::PI))?;
        let sinc = r#where(
            &eq(&sinc_phase, Array::from(0.0f32))?,
            Array::from(1.0f32),
            &sinc_phase.sin()?.divide(&sinc_phase)?,
        )?;
        let periodic =
            Array::from(1.0f32).subtract(&alpha.multiply(&sum)?.cos()?.multiply(&sinc)?)?;
        let activated = sum.divide(Array::from(2.0f32))?.add(
            &periodic.divide(
                &beta
                    .add(Array::from(1e-9f32))?
                    .multiply(Array::from(2.0f32))?,
            )?,
        )?;

        let down_padded = pad(
            &activated,
            &[(0, 0), (5, 6), (0, 0)],
            Array::from(0.0f32),
            Some(PadMode::Edge),
        )?;
        let down_filter = broadcast_to(&self.down_filter, &[self.channels, 12, 1])?;
        Ok(conv1d(&down_padded, &down_filter, 2, 0, 1, self.channels)?)
    }
}

struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations: Vec<Activation1d>,
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
        let mut activations = Vec::with_capacity(6);
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
            activations.push(Activation1d::load(
                weights,
                &format!("resblocks.{index}.activations.{}", layer * 2),
                channels,
            )?);
            activations.push(Activation1d::load(
                weights,
                &format!("resblocks.{index}.activations.{}", layer * 2 + 1),
                channels,
            )?);
        }
        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    fn forward(&mut self, input: &Array) -> Result<Array> {
        let mut output = input.clone();
        for layer in 0..3 {
            let activated = self.activations[layer * 2].forward(&output)?;
            let hidden = self.convs1[layer].forward(&activated)?;
            let activated = self.activations[layer * 2 + 1].forward(&hidden)?;
            output = output.add(&self.convs2[layer].forward(&activated)?)?;
        }
        Ok(output)
    }
}

struct ResampleUpsampler {
    scale_factor: i32,
    total_scale_factor: i32,
    channels: i32,
    input_insert_kernel: Array,
    source_insert_kernel: Array,
    julius_filter: Array,
    convolution_noise: Conv1d,
    convolution_after: Conv1d,
}

impl ResampleUpsampler {
    fn load(
        weights: &HashMap<String, Array>,
        index: usize,
        scale_factor: i32,
        total_scale_factor: i32,
        input_channels: i32,
        output_channels: i32,
    ) -> Result<Self> {
        Ok(Self {
            scale_factor,
            total_scale_factor,
            channels: input_channels,
            input_insert_kernel: Array::ones::<f32>(&[input_channels, 1, 1])?,
            source_insert_kernel: Array::ones::<f32>(&[INITIAL_CHANNELS, 1, 1])?,
            julius_filter: weight(weights, &format!("ups.{index}.julius_filter"))?,
            convolution_noise: conv1d_layer(
                weights,
                &format!("ups.{index}.convolution_noise"),
                INITIAL_CHANNELS,
                input_channels,
                7,
                3,
                1,
            )?,
            convolution_after: conv1d_layer(
                weights,
                &format!("ups.{index}.convolution_after"),
                input_channels,
                output_channels,
                1,
                0,
                1,
            )?,
        })
    }

    fn lowpass(&self, input: &Array) -> Result<Array> {
        let half_size = (self.julius_filter.shape()[1] - 1) / 2;
        let padded = pad(
            input,
            &[(0, 0), (half_size, half_size), (0, 0)],
            Array::from(0.0f32),
            Some(PadMode::Edge),
        )?;
        let filter = broadcast_to(
            &self.julius_filter,
            &[self.channels, self.julius_filter.shape()[1], 1],
        )?;
        Ok(conv1d(&padded, &filter, 1, 0, 1, self.channels)?)
    }

    fn forward(&mut self, input: &Array, source: &Array) -> Result<Array> {
        let source_upsampled = conv_transpose1d(
            source,
            &self.source_insert_kernel,
            self.total_scale_factor,
            0,
            1,
            self.total_scale_factor - 1,
            INITIAL_CHANNELS,
        )?;
        let source_filtered = self.convolution_noise.forward(&source_upsampled)?;
        let source_highpass = source_filtered.subtract(&self.lowpass(&source_filtered)?)?;

        let upsampled = conv_transpose1d(
            input,
            &self.input_insert_kernel,
            self.scale_factor,
            0,
            1,
            self.scale_factor - 1,
            self.channels,
        )?;
        let filtered = self.lowpass(&upsampled)?;
        self.convolution_after
            .forward(&filtered.add(&source_highpass)?)
            .map_err(Into::into)
    }
}

pub struct PupuFeatures {
    pub mel: Array,
    pub conv_pre: Array,
    pub resblocks: Vec<Array>,
    pub stages: Vec<Array>,
    pub output: Array,
}

pub struct PupuVocoder {
    mel: MelSpectrogram,
    conv_pre: Conv1d,
    upsamplers: Vec<ResampleUpsampler>,
    resblocks: Vec<ResBlock1>,
    activation_post: Activation1d,
    conv_post: Conv1d,
}

impl PupuVocoder {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let weights = Array::load_safetensors(path)?;
        let mel = MelSpectrogram::from_weights(&weights)?;
        let conv_pre = conv1d_layer(&weights, "conv_pre", 128, INITIAL_CHANNELS, 7, 3, 1)?;

        let mut upsamplers = Vec::with_capacity(5);
        let mut resblocks = Vec::with_capacity(15);
        let mut channels = INITIAL_CHANNELS;
        for stage in 0..5 {
            let output_channels = channels / 2;
            upsamplers.push(ResampleUpsampler::load(
                &weights,
                stage,
                UPSAMPLE_RATES[stage],
                TOTAL_UPSAMPLE_RATES[stage],
                channels,
                output_channels,
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
            resblocks,
            activation_post: Activation1d::load(&weights, "activation_post", channels)?,
            conv_post: conv1d_layer(&weights, "conv_post", channels, 1, 7, 3, 1)?,
        })
    }

    pub fn infer(&mut self, audio: &Array, target_length: usize) -> Result<Array> {
        let mel = self.mel.forward(audio)?;
        self.infer_mel(&mel, target_length)
    }

    pub fn infer_mel(&mut self, mel: &Array, target_length: usize) -> Result<Array> {
        Ok(self
            .infer_mel_internal(mel, false)?
            .output
            .index((.., ..target_length as i32, ..)))
    }

    pub fn infer_with_features(&mut self, audio: &Array) -> Result<PupuFeatures> {
        let mel = self.mel.forward(audio)?;
        self.infer_mel_internal(&mel, true)
    }

    fn infer_mel_internal(&mut self, mel: &Array, capture_features: bool) -> Result<PupuFeatures> {
        let mut hidden = self.conv_pre.forward(&mel)?;
        let conv_pre = hidden.clone();
        let source = hidden.clone();
        let mut resblocks = Vec::with_capacity(if capture_features { 15 } else { 0 });
        let mut stages = Vec::with_capacity(if capture_features { 5 } else { 0 });

        for stage in 0..5 {
            hidden = self.upsamplers[stage].forward(&hidden, &source)?;
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

        let activated = self.activation_post.forward(&hidden)?;
        let output = self.conv_post.forward(&activated)?;
        let output = maximum(&output, Array::from(-1.0f32))?;
        let output = minimum(&output, Array::from(1.0f32))?;
        output.eval()?;

        Ok(PupuFeatures {
            mel: mel.clone(),
            conv_pre,
            resblocks,
            stages,
            output,
        })
    }
}
