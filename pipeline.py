"""Standalone, disk-staged H3 inference with the validated precision settings."""
import json
import hashlib
import os
import resource
import time
from pathlib import Path
from settings import ROOT, RenderSettings

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_DISABLE_PROGRESS_BARS="1")


def configure_memory(settings: RenderSettings):
    """Apply the profile's allocator cap, including direct load_pipeline callers."""
    import torch
    capacity = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = settings.recipe["gpu_cap_gib"]
    fraction = min(cap / capacity, 1.0) if cap is not None else 1.0
    torch.cuda.set_per_process_memory_fraction(fraction)
    return capacity * fraction


def load_pipeline(settings: RenderSettings, timings=None):
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from download_models import COMPANIONS, TURBO_FILES
    from hybrid_checkpoint import PATH, SHA256
    from hybrid_lora import HybridTurboLoader, turbo_metadata
    from safe_attention import enable_memory_efficient_attention
    from selflift import attach_selflift

    receipt = json.loads((ROOT / "hybrid-download.json").read_text())
    if receipt.get("status") != "verified" or receipt.get("sha256") != SHA256:
        raise ValueError("Run download_models.py to verify the hybrid checkpoint")
    recipe = settings.recipe
    if settings.upscaler == "learned3d":
        from latent_upscaler import PATH as upscaler_path, verify_checkpoint
        if not upscaler_path.is_file():
            raise FileNotFoundError("Download first: uv run python latent_upscaler.py")
        verify_checkpoint(upscaler_path)
    configure_memory(settings)
    enable_memory_efficient_attention()
    offload = dict(offload_dtype="disk", offload_device="disk",
                   onload_dtype="disk", onload_device="disk",
                   preparing_dtype="disk", preparing_device="disk",
                   computation_device="cuda", computation_dtype=torch.float32)
    paths = [PATH, *(ROOT / "models/nf4" / name for name in COMPANIONS)]
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.float32, device="cuda",
        model_configs=[ModelConfig(path=str(path), **offload) for path in paths],
        processor_config=ModelConfig(path=str(ROOT / "models/h3/FL2VA/processor")),
        vram_limit=recipe["vram_limit"])
    video_decode, audio_decode = pipe.video_vae.decode_video, pipe.audio_vae.decode_audio

    def decode_video(latents, **kwargs):
        kwargs.update(dtype=torch.float32, tile_size=256)
        return video_decode(latents.float(), **kwargs)

    def decode_audio(latents, **kwargs):
        kwargs.update(dtype=torch.float32)
        return audio_decode(latents.float(), **kwargs)

    pipe.video_vae.decode_video, pipe.audio_vae.decode_audio = decode_video, decode_audio
    turbo_path = ROOT / "models/turbo" / TURBO_FILES[recipe["steps"]]
    if not turbo_path.is_file():
        raise FileNotFoundError(f"Download this adapter first: uv run python download_models.py --steps {recipe['steps']}")
    metadata = turbo_metadata(turbo_path)
    with turbo_path.open("rb") as stream:
        metadata.update(filename=turbo_path.name,
                        sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    pipe.lora_loader = HybridTurboLoader
    from diffsynth.core import load_state_dict
    # Preserve the adapter's BF16 values on CPU. The pinned scales are powers
    # of two; the upstream hotload path casts each pair to FP32 on demand.
    if metadata["scale"] not in (1.0, 0.0625):
        raise ValueError("CPU BF16 adapter storage requires the verified pinned Turbo scale")
    state = load_state_dict(str(turbo_path), torch_dtype=torch.bfloat16, device="cpu")
    pipe.load_lora(pipe.dit, state_dict=state, alpha=metadata["scale"])
    del state
    weights = [tensor for module in pipe.dit.modules()
               for attribute in ("lora_A_weights", "lora_B_weights")
               for tensor in getattr(module, attribute, [])]
    if not weights or not all(tensor.device.type == "cpu" for tensor in weights):
        raise ValueError("The under-8 profile requires all adapter weights to remain on CPU")
    metadata["storage_gib"] = sum(t.numel() * t.element_size() for t in weights) / 2**30
    metadata["storage"] = "cpu_bfloat16_streamed_as_float32"
    if timings is not None:
        pipe.video_vae.decode_video = timings.wrap(pipe.video_vae.decode_video, "video_decode")
        pipe.video_vae.encode_video = timings.wrap(pipe.video_vae.encode_video, "video_encode_lift")
        pipe.audio_vae.decode_audio = timings.wrap(pipe.audio_vae.decode_audio, "audio_decode")
        pipe.load_models_to_device = timings.wrap(pipe.load_models_to_device, "stage_models")
        pipe.cfg_guided_model_fn = timings.wrap(pipe.cfg_guided_model_fn, "dit_evaluation")
        original_runner = pipe.unit_runner
        def unit_runner(unit, *args, **kwargs):
            with timings.measure("unit:" + type(unit).__name__):
                return original_runner(unit, *args, **kwargs)
        pipe.unit_runner = unit_runner
    attach_selflift(pipe, height=recipe["target_height"], width=recipe["target_width"],
                    transition_step=recipe["transition_step"], rho=0.4,
                    seed=9174, vae_tile_size=256, upscaler=settings.upscaler)
    return pipe, metadata


def render(prompt: str, output: Path, settings: RenderSettings | None = None):
    """Render one scene and write an adjacent JSON receipt. Requires a CUDA GPU."""
    import torch
    from diffsynth.utils.data.audio_video import write_video_audio

    settings = settings or RenderSettings()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = settings.recipe
    effective_cap = configure_memory(settings)
    torch.cuda.reset_peak_memory_stats()
    from timings import StageTimings
    timings = StageTimings(output.with_suffix(".timings.json"))
    started = time.time()
    report = dict(status="started", settings=settings.model_dump(), prompt=prompt,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  recipe=recipe, effective_allocator_cap_gib=effective_cap,
                  git_commit=os.environ.get("H3_GIT_COMMIT", "unknown"),
                  width=recipe["target_width"], height=recipe["target_height"])
    if settings.upscaler == "learned3d":
        from latent_upscaler import REPO, REVISION, FILENAME, SHA256
        report["upscaler_model"] = dict(repo=REPO, revision=REVISION,
                                       filename=FILENAME, sha256=SHA256,
                                       computation_dtype="float32")
    receipt = output.with_suffix(".json")
    receipt.write_text(json.dumps(report, indent=2))
    try:
        with timings.measure("load_pipeline"):
            pipe, metadata = load_pipeline(settings, timings)
        with timings.measure("generate_total"):
            video, audio = pipe(prompt=prompt, width=recipe["width"], height=recipe["height"],
                            num_frames=settings.frames, num_inference_steps=recipe["steps"],
                            seed=settings.seed, flow_shift=6.0, audio_flow_shift=3.0, cfg_scale=1,
                            tiled=True, tile_size=256, tile_overlap=32)
        if not torch.isfinite(torch.as_tensor(audio)).all():
            raise ValueError("Generated audio contains non-finite samples")
        with timings.measure("export"):
            write_video_audio(video=video, audio=audio, output_path=str(output),
                          fps=24, audio_sample_rate=32000)
        report.update(status="success", video_file=output.name, turbo_metadata=metadata,
                      selflift_diagnostics=pipe.selflift_diagnostics,
                      stage_timings=timings.events,
                      peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.time() - started,
                      stage_timings=timings.events,
                      peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30,
                      peak_resident_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        receipt.write_text(json.dumps(report, indent=2))
    return output
