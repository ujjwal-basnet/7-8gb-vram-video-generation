"""Learned 2x H3 upscaling, followed by one low-noise refinement per window."""
import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path
import torch
from settings import ROOT, RenderSettings


def temporal_windows(length):
    """Use 39-frame windows with a frozen overlap, aligned to H3's five-token grid."""
    if length < 7 or length % 5 != 2:
        raise ValueError("Expected 5*n+2 H3 latent frames, with n >= 1")
    size = min(12, length)
    starts = list(range(0, length - size + 1, 10))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, start + size) for start in starts]


def read_video(source, frames=None):
    import av
    images = []
    with av.open(str(source)) as container:
        if not container.streams.audio or container.streams.video[0].average_rate != 24:
            raise ValueError("Use an H3 MP4 with audio at 24 fps")
        for frame in container.decode(video=0):
            images.append(frame.to_image())
            if len(images) == frames:
                break
            if len(images) > 362:
                raise ValueError("Use at most 362 frames, or pass --frames to select a prefix")
    if frames is not None and len(images) != frames:
        raise ValueError(f"Input has fewer than {frames} frames")
    if not images or any(dimension % 32 for dimension in images[0].size):
        raise ValueError("Input dimensions must be multiples of 32")
    return images


def refine(pipe, lifted, audio, positive, sigma, seed, timings, report, receipt):
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    noise = torch.randn(lifted.shape, device=pipe.device, dtype=torch.float32, generator=generator)
    result = torch.empty_like(lifted)
    pipe.load_models_to_device(pipe.in_iteration_models)
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    written = 0
    for start, stop in temporal_windows(lifted.shape[2]):
        base = lifted[:, :, start:stop]
        frames = (stop - start - 2) // 5 * 17 + 5
        audio_start = round(start / 5 * 17 / 24 * 40)
        audio_length = round(frames / 24 * 40)
        audio_chunk = audio[..., audio_start:audio_start + audio_length]
        if audio_chunk.shape[-1] < audio_length:
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, audio_length - audio_chunk.shape[-1]), mode="replicate")
        anchor = base.clone()
        overlap = max(0, written - start)
        anchor[:, :, :overlap] = result[:, :, start:written]
        mask = torch.ones_like(base[:, :1])
        mask[:, :, :overlap] = 0
        sample = ((1 - sigma) * base + sigma * noise[:, :, start:stop]) * mask + anchor * (1 - mask)
        shared = dict(video_latents=sample, audio_latents=audio_chunk,
                      input_latents_video=anchor, denoise_mask_video=mask)
        conditioning = {**positive, **builder.process(pipe, positive["prompt_embeds"], sample, audio_chunk,
                                                     text_token_tags=positive["text_token_tags"])}
        with timings.measure(f"refine:{start}:{stop}"):
            prediction, _ = pipe.cfg_guided_model_fn(pipe.model_fn, 1, shared, conditioning, {},
                dit=pipe.dit, controlnet=None, timestep_video=torch.tensor([sigma * 1000], device=pipe.device),
                timestep_audio=torch.tensor([0.], device=pipe.device))
        updated = (sample - sigma * prediction) * mask + anchor * (1 - mask)
        if not torch.isfinite(updated).all():
            raise ValueError("Refinement produced non-finite latents")
        result[:, :, start:stop] = updated
        written = stop
        report["windows"].append(dict(start=start, stop=stop, frozen_overlap=overlap))
        receipt.write_text(json.dumps(report, indent=2))
        del base, anchor, mask, sample, shared, conditioning, prediction, updated, audio_chunk
    return result


@torch.inference_mode()
def enhance(source, output, prompt, frames=None, steps=8, seed=9175, sigma=0.20):
    """Enhance a completed H3 clip; preserve its frame count, shot and original audio."""
    import numpy as np
    from pipeline import load_pipeline, configure_memory
    from latent_upscaler import learned_lift, PATH, verify_checkpoint
    from stream_video import export_stream
    from timings import StageTimings
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder

    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Choose a separate output path to preserve the input")
    if not 0 < sigma <= 0.30:
        raise ValueError("sigma must be greater than zero and at most 0.30")
    verify_checkpoint(PATH)
    images = read_video(source, frames)
    frames = len(images)
    settings = RenderSettings(mode="native", resolution="512x320", frames=frames, steps=steps, seed=seed)
    width, height = images[0].size
    if width > 640 or height > 384:
        raise ValueError("This memory profile supports input up to 640x384")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = configure_memory(settings)
    torch.cuda.reset_peak_memory_stats()
    receipt = output.with_suffix(".json")
    timings = StageTimings(output.with_suffix(".timings.json"))
    started = time.monotonic()
    report = dict(status="running", source=str(source), frames=frames, fps=24,
                  width=width * 2, height=height * 2, seed=seed, sigma=sigma,
                  method="finished_learned_2x_then_video_refine", windows=[],
                  allocator_cap_gib=cap, audio="original track retained",
                  gpu=torch.cuda.get_device_name(0), git_commit=os.environ.get("H3_GIT_COMMIT", "unknown"))
    receipt.write_text(json.dumps(report, indent=2))
    try:
        with timings.measure("load_pipeline"):
            pipe, metadata = load_pipeline(settings)
        with timings.measure("prompt_encoding"):
            positive = MiniMaxH3Unit_PromptEmbedder().process(pipe, prompt, height=height, width=width)
        pipe.load_models_to_device(["video_vae"])
        with timings.measure("encode_finished_video"):
            pixels = pipe.preprocess_video(images, torch_dtype=torch.float32, min_value=0, device=pipe.device)
            clean = pipe.video_vae.encode_video(pixels, dtype=torch.float32, tiled=True, tile_size=256, tile_overlap=32)
        del pixels, images
        pipe.load_models_to_device(["audio_vae"])
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(source), "-t", str(frames / 24),
                                       "-f", "f32le", "-ac", "2", "-ar", "32000", "pipe:1"])
        wave = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()).to(pipe.device)
        with timings.measure("audio_encode"):
            audio = pipe.audio_vae.encode_audio(wave, dtype=torch.float32)
        del wave, raw
        pipe.load_models_to_device([])
        with timings.measure("learned_lift"):
            lifted = learned_lift(clean, height * 2, width * 2)
        del clean
        result = refine(pipe, lifted, audio, positive, sigma, seed, timings, report, receipt)
        del lifted, audio, positive
        pipe.load_models_to_device(["video_vae"])
        with timings.measure("decode_and_export"):
            export_stream(pipe, result, source, output, frames)
        report.update(status="success", turbo_metadata=metadata, refinement_evaluations=len(report["windows"]))
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30,
                      peak_resident_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        receipt.write_text(json.dumps(report, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/dog-upscaled.mp4")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "prompts/dog-walking.txt")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int, choices=[4, 8], default=8)
    parser.add_argument("--seed", type=int, default=9175)
    parser.add_argument("--sigma", type=float, default=0.20)
    args = parser.parse_args()
    print(enhance(args.input, args.output, args.prompt_file.read_text().strip(),
                  frames=args.frames, steps=args.steps, seed=args.seed, sigma=args.sigma), flush=True)


if __name__ == "__main__":
    main()
