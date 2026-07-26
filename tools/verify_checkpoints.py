from pathlib import Path

import julius
import librosa
import numpy as np
import torch
from safetensors.torch import load_file


workspace = Path(__file__).resolve().parents[2]
checkpoint_root = workspace / "checkpoints"


def preprocessing_tensors(fmin, fmax):
    return {
        "preprocess.mel_basis": torch.from_numpy(
            librosa.filters.mel(
                sr=44_100,
                n_fft=2_048,
                n_mels=128,
                fmin=fmin,
                fmax=fmax,
                dtype=np.float32,
            ).T
        ).contiguous(),
        "preprocess.hann_window": torch.hann_window(
            2_048,
            periodic=True,
            dtype=torch.float32,
        ),
    }


def expected_weights(state, transpose_conv_transpose):
    expected = {}
    for key, tensor in state.items():
        if key.endswith(".weight_v"):
            prefix = key.removesuffix(".weight_v")
            weight = torch._weight_norm(
                tensor,
                state[f"{prefix}.weight_g"],
                dim=0,
            )
            if transpose_conv_transpose(prefix):
                weight = weight.permute(1, 2, 0)
            else:
                weight = weight.permute(0, 2, 1)
            expected[f"{prefix}.weight"] = weight.contiguous()
        elif key.endswith(".weight_g"):
            continue
        elif tensor.ndim == 3 and (
            key.endswith(".weight") or key.endswith(".filter")
        ):
            expected[key] = tensor.permute(0, 2, 1).contiguous()
        else:
            expected[key] = tensor.contiguous()
    return expected


def verify(name, expected, actual):
    assert expected.keys() == actual.keys(), (
        f"{name}: key mismatch: "
        f"missing={sorted(expected.keys() - actual.keys())}, "
        f"extra={sorted(actual.keys() - expected.keys())}"
    )
    maximum_error = 0.0
    for key in expected:
        assert expected[key].shape == actual[key].shape, (
            f"{name}/{key}: shape {actual[key].shape}, "
            f"expected {expected[key].shape}"
        )
        error = (expected[key] - actual[key]).abs().max().item()
        maximum_error = max(maximum_error, error)
        assert error == 0.0, f"{name}/{key}: max_abs_error={error}"
    print(f"{name}: {len(expected)} tensors, max_abs_error={maximum_error:.9g}")


pupu_source = load_file(
    checkpoint_root
    / "pupuvocoder_large"
    / "checkpoint"
    / "epoch-0026_step-2315282_loss-46.095750"
    / "model.safetensors",
    device="cpu",
)
pupu_expected = expected_weights(pupu_source, lambda _: False)
pupu_expected.update(preprocessing_tensors(0, 22_050))
for index, rate in enumerate((8, 8, 2, 2, 2)):
    lowpass = julius.LowPassFilter(0.5 / rate)
    pupu_expected[f"ups.{index}.julius_filter"] = (
        lowpass._lowpasses.filters.permute(0, 2, 1).contiguous()
    )
pupu_actual = load_file(
    checkpoint_root / "pupuvocoder_large" / "model.mlx.safetensors",
    device="cpu",
)
verify("pupuvocoder_large", pupu_expected, pupu_actual)

pc_source = torch.load(
    checkpoint_root
    / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    / "model.ckpt",
    map_location="cpu",
    weights_only=True,
)["generator"]
pc_expected = expected_weights(
    pc_source,
    lambda prefix: prefix.startswith("ups."),
)
pc_expected.update(preprocessing_tensors(40, 16_000))
pc_actual = load_file(
    checkpoint_root
    / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    / "model.mlx.safetensors",
    device="cpu",
)
verify("pc_nsf_hifigan", pc_expected, pc_actual)
