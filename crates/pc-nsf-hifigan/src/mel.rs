use std::collections::HashMap;

use anyhow::{Context, Result};
use mlx_rs::Array;
use mlx_rs::fft::rfft;
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};
use mlx_rs::ops::{as_strided, concatenate_axis, maximum, square};

use crate::HOP_SIZE;

const FFT_SIZE: i32 = 2_048;
const PAD: i32 = (FFT_SIZE - HOP_SIZE as i32) / 2;

pub struct MelSpectrogram {
    mel_basis: Array,
    hann_window: Array,
}

impl MelSpectrogram {
    pub fn from_weights(weights: &HashMap<String, Array>) -> Result<Self> {
        Ok(Self {
            mel_basis: weights
                .get("preprocess.mel_basis")
                .context("缺少 preprocess.mel_basis")?
                .clone(),
            hann_window: weights
                .get("preprocess.hann_window")
                .context("缺少 preprocess.hann_window")?
                .clone(),
        })
    }

    pub fn forward(&self, audio: &Array) -> Result<Array> {
        let batch = audio.shape()[0];
        let length = audio.shape()[1];
        let left = audio
            .index((.., 1..PAD + 1))
            .index((.., (..).stride_by(-1)));
        let right = audio
            .index((.., length - PAD - 1..length - 1))
            .index((.., (..).stride_by(-1)));
        let padded = concatenate_axis(&[&left, audio, &right], 1)?;
        let padded_length = padded.shape()[1];
        let frames = (padded_length - FFT_SIZE) / HOP_SIZE as i32 + 1;

        let framed = as_strided(
            &padded,
            &[batch, frames, FFT_SIZE],
            &[padded_length as i64, HOP_SIZE as i64, 1],
            0,
        )?;
        let windowed = framed.multiply(&self.hann_window.reshape(&[1, 1, FFT_SIZE])?)?;
        let spectrum = rfft(&windowed, FFT_SIZE, None)?;
        let magnitude = square(&spectrum.real()?)?
            .add(&square(&spectrum.imag()?)?)?
            .sqrt()?;
        let mel = magnitude.matmul(&self.mel_basis)?;
        Ok(maximum(&mel, Array::from(1e-5f32))?.log()?)
    }
}
