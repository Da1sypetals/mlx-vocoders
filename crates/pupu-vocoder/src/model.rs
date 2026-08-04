use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use mlx_rs::Array;
use mlx_rs::builder::Builder;
use mlx_rs::module::Module;
use mlx_rs::nn::{Conv1d, Conv1dBuilder};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, maximum, minimum};

use crate::MelSpectrogram;
use crate::metal::{MetalActivationKernel, MetalResampleKernel};

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
    kernel: MetalActivationKernel,
}

impl Activation1d {
    fn load(weights: &HashMap<String, Array>, prefix: &str) -> Result<Self> {
        Ok(Self {
            alpha: weight(weights, &format!("{prefix}.act.alpha"))?,
            beta: weight(weights, &format!("{prefix}.act.beta"))?,
            up_filter: weight(weights, &format!("{prefix}.upsample.filter"))?,
            down_filter: weight(weights, &format!("{prefix}.downsample.lowpass.filter"))?,
            kernel: MetalActivationKernel::new()?,
        })
    }

    fn forward(&self, input: &Array) -> Result<Array> {
        self.kernel.forward(
            input,
            &self.alpha,
            &self.beta,
            &self.up_filter,
            &self.down_filter,
        )
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
            )?);
            activations.push(Activation1d::load(
                weights,
                &format!("resblocks.{index}.activations.{}", layer * 2 + 1),
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
    source_head_weight: Array,
    source_tail_weight: Array,
    source_bias: Array,
    julius_filter: Array,
    kernel: MetalResampleKernel,
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
        let convolution_noise = conv1d_layer(
            weights,
            &format!("ups.{index}.convolution_noise"),
            INITIAL_CHANNELS,
            input_channels,
            7,
            3,
            1,
        )?;
        let source_head_weight = concatenate_axis(
            &[
                convolution_noise.weight.value.index((.., 3..4, ..)),
                convolution_noise.weight.value.index((.., 2..3, ..)),
                convolution_noise.weight.value.index((.., 1..2, ..)),
                convolution_noise.weight.value.index((.., 0..1, ..)),
            ],
            1,
        )?
        .transpose_axes(&[2, 1, 0])?
        .reshape(&[INITIAL_CHANNELS, 4 * input_channels])?;
        let source_tail_weight = concatenate_axis(
            &[
                convolution_noise.weight.value.index((.., 6..7, ..)),
                convolution_noise.weight.value.index((.., 5..6, ..)),
                convolution_noise.weight.value.index((.., 4..5, ..)),
            ],
            1,
        )?
        .transpose_axes(&[2, 1, 0])?
        .reshape(&[INITIAL_CHANNELS, 3 * input_channels])?;
        source_head_weight.eval()?;
        source_tail_weight.eval()?;
        let source_bias = convolution_noise
            .bias
            .value
            .context("Pupu convolution_noise 缺少 bias")?;
        Ok(Self {
            scale_factor,
            total_scale_factor,
            channels: input_channels,
            source_head_weight,
            source_tail_weight,
            source_bias,
            julius_filter: weight(weights, &format!("ups.{index}.julius_filter"))?,
            kernel: MetalResampleKernel::new()?,
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

    fn forward(&mut self, input: &Array, source: &Array) -> Result<Array> {
        let batch = source.shape()[0];
        let frames = source.shape()[1];
        let head =
            source
                .matmul(&self.source_head_weight)?
                .reshape(&[batch, frames, 4, self.channels])?;
        let final_source_frame = Array::zeros::<f32>(&[batch, 1, INITIAL_CHANNELS])?;
        let shifted_source =
            concatenate_axis(&[&source.index((.., 1.., ..)), &final_source_frame], 1)?;
        let tail = shifted_source.matmul(&self.source_tail_weight)?.reshape(&[
            batch,
            frames,
            3,
            self.channels,
        ])?;
        let empty_phases =
            Array::zeros::<f32>(&[batch, frames, self.total_scale_factor - 7, self.channels])?;
        let source_filtered = concatenate_axis(&[&head, &empty_phases, &tail], 2)?
            .reshape(&[batch, frames * self.total_scale_factor, self.channels])?
            .add(&self.source_bias)?;
        let combined = self.kernel.forward(
            input,
            &source_filtered,
            &self.julius_filter,
            self.scale_factor,
        )?;
        self.convolution_after
            .forward(&combined)
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
            activation_post: Activation1d::load(&weights, "activation_post")?,
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
        let mut hidden = self.conv_pre.forward(mel)?;
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
