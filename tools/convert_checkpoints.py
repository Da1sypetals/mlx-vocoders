from pathlib import Path

import julius
import librosa
import numpy as np
import torch
from safetensors.torch import load_file, save_file


workspace = Path(__file__).resolve().parents[2]
checkpoint_root = workspace / "checkpoints"


def preprocessing_tensors(fmin, fmax):
    mel_basis = librosa.filters.mel(
        sr=44_100,
        n_fft=2_048,
        n_mels=128,
        fmin=fmin,
        fmax=fmax,
        dtype=np.float32,
    )
    return {
        "preprocess.mel_basis": torch.from_numpy(mel_basis.T).contiguous(),
        "preprocess.hann_window": torch.hann_window(
            2_048,
            periodic=True,
            dtype=torch.float32,
        ),
    }


def remove_weight_norm(state, transpose_conv_transpose):
    converted = {}
    consumed = set()
    for key, tensor in state.items():
        if key in consumed:
            continue
        if key.endswith(".weight_v"):
            prefix = key.removesuffix(".weight_v")
            weight_g_key = f"{prefix}.weight_g"
            weight = torch._weight_norm(tensor, state[weight_g_key], dim=0)
            consumed.add(weight_g_key)
            if transpose_conv_transpose(prefix):
                weight = weight.permute(1, 2, 0)
            else:
                weight = weight.permute(0, 2, 1)
            converted[f"{prefix}.weight"] = weight.contiguous()
        elif key.endswith(".weight_g"):
            continue
        elif tensor.ndim == 3 and (
            key.endswith(".weight") or key.endswith(".filter")
        ):
            converted[key] = tensor.permute(0, 2, 1).contiguous()
        else:
            converted[key] = tensor.contiguous()
    return converted


pupu_source = (
    checkpoint_root
    / "pupuvocoder_large"
    / "checkpoint"
    / "epoch-0026_step-2315282_loss-46.095750"
    / "model.safetensors"
)
pupu_target = checkpoint_root / "pupuvocoder_large" / "model.mlx.safetensors"
pupu_state = load_file(pupu_source, device="cpu")
pupu_converted = remove_weight_norm(pupu_state, lambda _: False)
pupu_converted.update(preprocessing_tensors(0, 22_050))
for index, rate in enumerate((8, 8, 2, 2, 2)):
    lowpass = julius.LowPassFilter(0.5 / rate)
    pupu_converted[f"ups.{index}.julius_filter"] = (
        lowpass._lowpasses.filters.permute(0, 2, 1).contiguous()
    )
save_file(
    pupu_converted,
    pupu_target,
    metadata={
        "format": "mlx",
        "layout": "NLC",
        "model": "pupuvocoder_large",
        "source": str(pupu_source.relative_to(workspace)),
    },
)

pc_source = (
    checkpoint_root
    / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    / "model.ckpt"
)
pc_target = (
    checkpoint_root
    / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    / "model.mlx.safetensors"
)
pc_state = torch.load(pc_source, map_location="cpu", weights_only=True)["generator"]
pc_converted = remove_weight_norm(
    pc_state,
    lambda prefix: prefix.startswith("ups."),
)
pc_converted.update(preprocessing_tensors(40, 16_000))
save_file(
    pc_converted,
    pc_target,
    metadata={
        "format": "mlx",
        "layout": "NLC",
        "model": "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
        "source": str(pc_source.relative_to(workspace)),
    },
)

print(pupu_target)
print(pc_target)
