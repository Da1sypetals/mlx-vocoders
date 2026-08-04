mod audio;
mod mel;
mod metal;
mod model;

pub use audio::{load_mono_audio, save_pcm16_wav};
pub use mel::MelSpectrogram;
pub use model::{PupuFeatures, PupuVocoder};

pub const SAMPLE_RATE: u32 = 44_100;
pub const HOP_SIZE: usize = 512;
pub const MEL_BINS: usize = 128;
