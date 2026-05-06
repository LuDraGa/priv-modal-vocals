"""Compatibility patches for current Dia2 runtime assumptions."""


class _CudnnConvCompat:
    """Minimal object expected by Dia2's TF32 setup path."""

    fp32_precision = "tf32"


def patch_torch_cudnn_conv() -> None:
    """Provide torch.backends.cudnn.conv when the installed Torch lacks it."""
    import torch

    if not hasattr(torch.backends.cudnn, "conv"):
        torch.backends.cudnn.conv = _CudnnConvCompat()
