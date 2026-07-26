import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import torch
from safetensors.torch import load_file


workspace = Path(__file__).resolve().parents[2]
upstream = workspace / "upstream"
checkpoint_root = workspace / "checkpoints"


def save_nlc(path, tensor):
    np.save(path, tensor.detach().permute(0, 2, 1).contiguous().numpy())


def load_audio(path, samples):
    audio, _ = librosa.load(path, sr=44_100, mono=True)
    if samples is not None:
        audio = audio[:samples]
    return torch.from_numpy(audio).reshape(1, -1)


def pupu_config():
    preprocess = SimpleNamespace(
        n_mel=128,
        sample_rate=44_100,
        n_fft=2_048,
        win_size=2_048,
        hop_size=512,
        fmin=0,
        fmax=22_050,
    )
    pupuvocoder = SimpleNamespace(
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel=1_536,
        upsample_kernel_sizes=[16, 16, 4, 4, 4],
        upsample_rates=[8, 8, 2, 2, 2],
    )
    return SimpleNamespace(
        preprocess=preprocess,
        model=SimpleNamespace(pupuvocoder=pupuvocoder),
    )


def run_pupu(audio_path, output_dir, samples):
    source_root = upstream / "AliasingFreeNeuralAudioSynthesis"
    sys.path.insert(0, str(source_root))
    from models.vocoders.gan.generator.pupuvocoder import PupuVocoder
    from utils.mel import extract_mel_features

    cfg = pupu_config()
    model = PupuVocoder(cfg)
    source_checkpoint = (
        checkpoint_root
        / "pupuvocoder_large"
        / "checkpoint"
        / "epoch-0026_step-2315282_loss-46.095750"
        / "model.safetensors"
    )
    model.load_state_dict(load_file(source_checkpoint, device="cpu"))
    model.eval()

    audio = load_audio(audio_path, samples)
    aligned_length = ((audio.shape[1] + 511) // 512) * 512
    audio_aligned = torch.nn.functional.pad(
        audio,
        (0, aligned_length - audio.shape[1]),
    )
    mel = extract_mel_features(audio_aligned, cfg.preprocess).unsqueeze(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "audio.npy", audio.numpy())
    save_nlc(output_dir / "mel.npy", mel)
    with torch.inference_mode():
        hidden = model.conv_pre(mel)
        save_nlc(output_dir / "conv_pre.npy", hidden)
        source = hidden
        for stage in range(model.num_upsamples):
            hidden = model.ups[stage](hidden, source, model.upps[stage])
            stage_outputs = [
                model.resblocks[stage * model.num_kernels + kernel](hidden)
                for kernel in range(model.num_kernels)
            ]
            for kernel, stage_output in enumerate(stage_outputs):
                save_nlc(
                    output_dir / f"stage_{stage}_resblock_{kernel}.npy",
                    stage_output,
                )
            hidden = torch.stack(stage_outputs).sum(dim=0) / model.num_kernels
            save_nlc(output_dir / f"stage_{stage}.npy", hidden)
        hidden = model.activation_post(hidden)
        output = torch.clamp(model.conv_post(hidden), min=-1.0, max=1.0)
        save_nlc(output_dir / "output.npy", output)


def run_pc(audio_path, rust_dir, output_dir, samples):
    source_root = upstream / "DiffSinger"
    sys.path.insert(0, str(source_root))
    from modules.nsf_hifigan.env import AttrDict
    from modules.nsf_hifigan.models import Generator
    from modules.nsf_hifigan.nvSTFT import STFT

    checkpoint_dir = (
        checkpoint_root / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    )
    with open(checkpoint_dir / "config.json") as file:
        config = AttrDict(json.load(file))
    model = Generator(config)
    checkpoint = torch.load(
        checkpoint_dir / "model.ckpt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    model.remove_weight_norm()

    audio = load_audio(audio_path, samples)
    mel_extractor = STFT(
        sr=44_100,
        n_mels=128,
        n_fft=2_048,
        win_size=2_048,
        hop_length=512,
        fmin=40,
        fmax=16_000,
        device=torch.device("cpu"),
    )
    mel = mel_extractor.get_mel(audio, center=False)
    f0 = torch.from_numpy(np.load(rust_dir / "f0.npy"))
    assert f0.shape == (1, mel.shape[2]), (f0.shape, mel.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "audio.npy", audio.numpy())
    save_nlc(output_dir / "mel.npy", mel)
    np.save(output_dir / "f0.npy", f0.numpy())
    with torch.inference_mode():
        source = model.fastsinegen(f0)
        save_nlc(output_dir / "source.npy", source)
        hidden = model.conv_pre(mel)
        save_nlc(output_dir / "conv_pre.npy", hidden)
        for stage in range(model.num_upsamples):
            hidden = model.ups[stage](
                torch.nn.functional.leaky_relu(hidden, 0.1)
            )
            if stage == 1:
                hidden = hidden + model.source_conv(source)
            stage_outputs = [
                model.resblocks[stage * model.num_kernels + kernel](hidden)
                for kernel in range(model.num_kernels)
            ]
            for kernel, stage_output in enumerate(stage_outputs):
                save_nlc(
                    output_dir / f"stage_{stage}_resblock_{kernel}.npy",
                    stage_output,
                )
            hidden = torch.stack(stage_outputs).sum(dim=0) / model.num_kernels
            save_nlc(output_dir / f"stage_{stage}.npy", hidden)
        hidden = torch.nn.functional.leaky_relu(hidden)
        output = torch.tanh(model.conv_post(hidden))
        save_nlc(output_dir / "output.npy", output)


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=("pupu", "pc"))
parser.add_argument("audio", type=Path)
parser.add_argument("rust_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--samples", type=int)
arguments = parser.parse_args()

if arguments.model == "pupu":
    run_pupu(arguments.audio, arguments.output_dir, arguments.samples)
else:
    run_pc(
        arguments.audio,
        arguments.rust_dir,
        arguments.output_dir,
        arguments.samples,
    )
