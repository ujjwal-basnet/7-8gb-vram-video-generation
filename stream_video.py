"""Decode H3 in temporal chunks and send frames directly to FFmpeg.

The overlap/blending schedule follows DiffSynth-Studio's pinned H3 VAE
decoder (Apache-2.0; see licenses/diffsynth-APACHE-2.0.txt).
"""
import subprocess


def decoded_chunks(vae, latents):
    """Yield RGB [1,3,T,H,W] chunks with the stock decoder's exact frame order."""
    import torch
    from diffsynth.models.minimax_h3_video_vae import _VIDEO_LATENTS_MEAN, _VIDEO_LATENTS_STD

    if latents.shape[2] < 7 or latents.shape[2] % 5 != 2:
        raise ValueError("H3 decoding requires 5*n+2 latent frames, with n >= 1")
    mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=latents.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(_VIDEO_LATENTS_STD, device=latents.device).view(1, -1, 1, 1, 1)
    raw = latents.float() * std + mean
    vae.tile_size, vae.tile_overlap_min = 256, 32
    carry = None
    chunk_size = vae.tokens_chunk_size
    frame_split = chunk_size * vae.vae_ratio_t
    for start in range(0, raw.shape[2] - vae.token_overlap, chunk_size):
        pixels = vae.tiled_decode(raw[:, :, start:start + chunk_size + vae.token_overlap])
        main = pixels[:, :, :frame_split][:, :, vae.frame_pre_padding:]
        if carry is not None:
            main = vae.blend(carry, main, vae.frame_overlap, dim=-3)
        carry = pixels[:, :, frame_split:][:, :, vae.frame_pre_padding:].contiguous()
        yield vae.processor.revert_tensor(main.float())
        del main, pixels
    yield vae.processor.revert_tensor(carry.float())


def export_stream(pipe, latents, source, output, frames):
    """Keep only one decoded chunk in memory and retain the source audio track."""
    import torch
    height, width = (dimension * 16 for dimension in latents.shape[-2:])
    silent = output.with_name(output.stem + ".silent.mp4")
    partial = output.with_name(output.stem + ".partial.mp4")
    command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-framerate", "24", "-i", "pipe:0",
               "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
               "-an", str(silent)]
    written = 0
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for chunk in decoded_chunks(pipe.video_vae, latents):
            if not torch.isfinite(chunk).all():
                raise ValueError("Decoded video contains non-finite pixels")
            images = pipe.vae_output_to_video(chunk, min_value=0, max_value=1)
            for image in images:
                encoder.stdin.write(image.tobytes())
            written += len(images)
            print(f"DECODE {written}/{frames} frames", flush=True)
            del chunk, images
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg video encoding failed")
        if written != frames:
            raise ValueError(f"Expected {frames} frames, decoded {written}")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(source),
                        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-t", str(frames / 24),
                        "-movflags", "+faststart", str(partial)], check=True)
        partial.replace(output)
    finally:
        if encoder.stdin and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        silent.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
