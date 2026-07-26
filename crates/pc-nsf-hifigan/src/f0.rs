use std::path::Path;

use anyhow::Result;
use fcpe_mlxrs::{
    CFNaiveMelPE, build_hann_window, build_mel_filterbank, postprocess_f0, resample_audio,
    wav_to_mel,
};
use mlx_rs::Array;
use mlx_rs::ops::indexing::{IndexOp, take_along_axis};
use mlx_rs::ops::{broadcast_to, concatenate_axis, minimum};

use crate::{HOP_SIZE, SAMPLE_RATE};

const FCPE_SAMPLE_RATE: usize = 16_000;
const FCPE_HOP_SIZE: f32 = 160.0;

pub struct FcpeEstimator {
    model: CFNaiveMelPE,
    mel_basis: Array,
    hann_window: Array,
}

impl FcpeEstimator {
    pub fn load(path: impl AsRef<Path>) -> Self {
        Self {
            model: CFNaiveMelPE::load(path),
            mel_basis: build_mel_filterbank(16_000.0, 1_024, 128, 0.0, 8_000.0),
            hann_window: build_hann_window(1_024),
        }
    }

    pub fn estimate(&mut self, audio: &Array, target_frames: i32) -> Result<Array> {
        audio.eval()?;
        let resampled = resample_audio(
            audio.as_slice::<f32>(),
            SAMPLE_RATE as usize,
            FCPE_SAMPLE_RATE,
        );
        let resampled_length = resampled.len() as i32;
        let resampled = Array::from_slice(&resampled, &[1, resampled_length]);
        let mel = wav_to_mel(&resampled, &self.mel_basis, &self.hann_window);
        let raw_f0 = self.model.infer(&mel, "local_argmax", 0.006);
        let (f0, _) = postprocess_f0(&raw_f0, self.model.f0_min, Some(self.model.f0_max), true);

        let source_frames = f0.shape()[1];
        let target_step = HOP_SIZE as f32 / SAMPLE_RATE as f32;
        let source_step = FCPE_HOP_SIZE / FCPE_SAMPLE_RATE as f32;
        let available_frames =
            (((source_frames - 1) as f32 * source_step) / target_step).ceil() as i32;
        let interpolation_frames = target_frames.min(available_frames);
        let positions = Array::arange::<_, f32>(None, interpolation_frames, None)?
            .multiply(Array::from(target_step / source_step))?;
        let left = positions.floor()?.as_type::<i32>()?;
        let right = minimum(
            &left.add(Array::from(1i32))?,
            Array::from(source_frames - 1),
        )?;
        let fraction = positions.subtract(&left.as_type::<f32>()?)?;
        let squeezed = f0.reshape(&[1, source_frames])?;
        let left_values = take_along_axis(&squeezed, &left.reshape(&[1, -1])?, 1)?;
        let right_values = take_along_axis(&squeezed, &right.reshape(&[1, -1])?, 1)?;
        let interpolated = left_values.add(
            &right_values
                .subtract(&left_values)?
                .multiply(&fraction.reshape(&[1, -1])?)?,
        )?;

        if interpolation_frames == target_frames {
            Ok(interpolated)
        } else {
            let last = interpolated.index((.., -1i32)).reshape(&[1, 1])?;
            let tail = broadcast_to(&last, &[1, target_frames - interpolation_frames])?;
            Ok(concatenate_axis(&[&interpolated, &tail], 1)?)
        }
    }
}
