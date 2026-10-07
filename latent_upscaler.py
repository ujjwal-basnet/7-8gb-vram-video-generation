"""Learned H3 lift with the released Conv v1 node's latent scaling contract."""
import hashlib
from pathlib import Path
from settings import ROOT

REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
REVISION = "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"
FILENAME = "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
SHA256 = "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6"
PATH = ROOT / "models/upscaler" / FILENAME


def verify_checkpoint(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError("H3 upscaler checkpoint hash differs from the pinned release")
    return Path(path)


def download():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, FILENAME, revision=REVISION,
                           local_dir=ROOT / "models/upscaler")
    return verify_checkpoint(path)


def learned_lift(clean, height, width):
    """Run the published Conv v1 network once, then release its GPU weights.

    Inputs/outputs are codec-normalized H3 latents. Conv v1 still requires
    the released node's additional per-channel affine transform around the
    network. Omitting it produces striped, invalid latents. Audio never enters.
    The caller must unload H3 before this function and restage it afterward.
    """
    import torch
    from safetensors.torch import load_file
    from latent_upscaler_network import LatentResizer3D
    from diffsynth.models.minimax_h3_video_vae import _VIDEO_LATENTS_MEAN, _VIDEO_LATENTS_STD
    if clean.ndim != 5 or clean.shape[1] != 24:
        raise ValueError("Expected normalized H3 video latents [B,24,T,H,W]")
    if height % 32 or width % 32:
        raise ValueError("H3 target dimensions must be multiples of 32")
    ratios = (height / (clean.shape[-2] * 16), width / (clean.shape[-1] * 16))
    if not all(1 <= ratio <= 4 for ratio in ratios):
        raise ValueError("The released upscaler supports spatial scales from 1x to 4x")
    if not PATH.is_file():
        raise FileNotFoundError("Download first: uv run python latent_upscaler.py")
    verify_checkpoint(PATH)
    with torch.device("meta"):
        model = LatentResizer3D(attn=False)
    model.load_state_dict(load_file(str(PATH), device="cpu"), strict=True, assign=True)
    model = model.to(device=clean.device, dtype=torch.float32).eval().requires_grad_(False)
    mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=clean.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(_VIDEO_LATENTS_STD, device=clean.device).view(1, -1, 1, 1, 1)
    try:
        with torch.inference_mode():
            output = model((clean.float() - mean) / std, scale=sum(ratios) / 2,
                           target_size=(clean.shape[2], height // 16, width // 16),
                           enable_chunking=True)
            output = output * std + mean
        if not torch.isfinite(output).all():
            raise ValueError("Learned H3 lift produced non-finite video latents")
        return output.to(dtype=clean.dtype)
    finally:
        del model
        if clean.is_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    print(download())
