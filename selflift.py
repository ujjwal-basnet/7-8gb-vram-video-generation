"""Experimental text-to-video SelfLift-zero adaptation for DiffSynth H3.

Independent orchestration of paired latent/pixel lifts and artifact correction.
The upstream VAE handles its own pixel and latent normalization.
"""
import torch
import torch.nn.functional as F
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder


def artifact_correct(direct, anchor, rho=0.4, wmin=0.5, wmax=1.0):
    """SelfLift-zero Eqs. 7–9: sample-wise selected-region adaptive weights."""
    if direct.shape != anchor.shape or not 0 <= rho <= 1 or not 0 <= wmin <= wmax <= 1:
        raise ValueError("Invalid paired lifts or correction parameters")
    if rho == 0:
        return direct
    delta = anchor.float() - direct.float()
    risk = delta.abs().mean(dim=1, keepdim=True)
    axes = tuple(range(1, risk.ndim))
    threshold = torch.quantile(risk.flatten(1), 1-rho, dim=1).reshape((-1,)+(1,)*(risk.ndim-1))
    mask = risk >= threshold
    minimum = risk.masked_fill(~mask, float("inf")).amin(dim=axes, keepdim=True)
    maximum = risk.masked_fill(~mask, float("-inf")).amax(dim=axes, keepdim=True)
    scaled = ((risk-minimum)/(maximum-minimum+1e-8)).clamp(0, 1)
    weights = mask * (wmin+(wmax-wmin)*scaled)
    return direct.float() + weights * delta


def pixel_anchor(pipe, clean, height, width, tile_size=256):
    pipe.load_models_to_device(["video_vae"])
    pixels = pipe.video_vae.decode_video(clean, dtype=torch.float32,
                                         tiled=True, tile_size=tile_size, tile_overlap=32)
    if not torch.isfinite(pixels).all():
        raise ValueError("VAE decoded non-finite pixels during SelfLift")
    batch, channels, frames, _, _ = pixels.shape
    images = pixels.permute(0, 2, 1, 3, 4).reshape(batch*frames, channels, *pixels.shape[-2:])
    images = F.interpolate(images.float(), size=(height, width), mode="bicubic",
                           align_corners=False, antialias=True).clamp(0, 1)
    pixels = images.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    anchor = pipe.video_vae.encode_video(pixels, dtype=torch.float32,
                                         tiled=True, tile_size=tile_size, tile_overlap=32)
    if not torch.isfinite(anchor).all():
        raise ValueError("VAE encoded a non-finite SelfLift anchor")
    pipe.load_models_to_device(pipe.in_iteration_models)
    return anchor


def attach_selflift(pipe, height=384, width=640, transition_step=2, rho=0.4, seed=9174,
                    vae_tile_size=256, upscaler="interpolate"):
    """Install a single-use lift before transition_step, keeping the audio trajectory.

    Supports the unconditioned text-to-video experiment with CFG=1 only.
    H3's backend prediction is -velocity and scheduler sigma is timestep/1000.
    """
    if height % 32 or width % 32 or min(height, width) < 32 or transition_step < 1:
        raise ValueError("H3 dimensions must be multiples of 32 and transition_step must be positive")
    original = getattr(pipe, "_selflift_original", pipe.cfg_guided_model_fn)
    pipe._selflift_original = original
    state = {"step": 0}
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    builder = MiniMaxH3Unit_PackedSequenceBuilder()

    def predict(model_fn, cfg_scale, shared, positive, negative, **kwargs):
        if cfg_scale != 1 or any(shared.get(k) is not None for k in
                                ("keyframes", "references", "control_video", "retake_video")):
            raise ValueError("This experiment supports text-to-video with CFG=1 only.")
        sigma = float(kwargs["timestep_video"].item()) / 1000
        if state["step"] == 0 or sigma == 1.0:
            if transition_step >= len(pipe.scheduler.timesteps):
                raise ValueError("SelfLift needs at least one remaining high-resolution evaluation")
            state.clear()
            state["step"] = 0
            generator.manual_seed(seed)
        if state["step"] == transition_step:
            print("SELFLIFT paired VAE/latent lift", flush=True)
            clean = state["latents"].float() - state["sigma"] * state["prediction"].float()
            if not torch.isfinite(clean).all():
                raise ValueError("Denoiser clean estimate is non-finite before SelfLift")
            if upscaler == "learned3d":
                from latent_upscaler import learned_lift
                pipe.load_models_to_device([])
                direct = learned_lift(clean, height, width)
            else:
                direct = F.interpolate(clean, size=(clean.shape[2], height//16, width//16), mode="nearest")
            anchor = pixel_anchor(pipe, clean, height, width, tile_size=vae_tile_size)
            if direct.shape != anchor.shape:
                raise ValueError(f"Paired lift shapes differ: {direct.shape} vs {anchor.shape}")
            corrected = artifact_correct(direct, anchor, rho=rho)
            pipe.selflift_diagnostics = {
                name: {"mean": float(value.float().mean()),
                       "std": float(value.float().std()),
                       "max_abs": float(value.float().abs().max())}
                for name, value in {"clean": clean, "direct": direct,
                                    "anchor": anchor, "corrected": corrected}.items()
            }
            print("SELFLIFT diagnostics", pipe.selflift_diagnostics, flush=True)
            noise = torch.randn(corrected.shape, generator=generator, device=pipe.device, dtype=pipe.torch_dtype)
            shared["video_latents"] = ((1-sigma)*corrected + sigma*noise).to(pipe.torch_dtype)
            shared.update(height=height, width=width)
            positive.update(builder.process(pipe, prompt_embeds=positive["prompt_embeds"],
                text_token_tags=positive["text_token_tags"],
                video_latents=shared["video_latents"], audio_latents=shared["audio_latents"]))
            if not torch.isfinite(shared["video_latents"]).all():
                raise ValueError("SelfLift produced non-finite latents")
            print("SELFLIFT resume", tuple(shared["video_latents"].shape), "sigma", sigma, flush=True)
        latents = shared["video_latents"].detach().clone()
        prediction = original(model_fn, cfg_scale, shared, positive, negative, **kwargs)
        if not all(bool(torch.isfinite(value).all()) for value in prediction):
            raise ValueError(f"Denoiser prediction is non-finite at step {state['step']}")
        state.update(step=state["step"]+1, latents=latents,
                     sigma=sigma, prediction=prediction[0].detach())
        return prediction

    pipe.cfg_guided_model_fn = predict
    return pipe
