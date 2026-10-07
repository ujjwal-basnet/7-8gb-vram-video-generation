# 7–8 GB VRAM Video Generation

Python/PyTorch video generation with H3 hybrid INT8, Turbo and optional experimental SelfLift-zero. No ComfyUI runtime. The default prompt is one dog walking with a steady camera and simple background. `prompts/lantern.txt` is another simple prompt.

Requires Linux, Python 3.12, a CUDA GPU, `ffmpeg`, [uv](https://docs.astral.sh/uv/getting-started/installation/), around 60 GB disk and preferably 16 GB system RAM. The eight-step native dog sample on a Colab T4 peaked at 4.50 GiB sampled GPU memory and 9.37 GiB process RAM. The T4 has 16 GB VRAM; these runs use a 6.5-GiB allocator cap. A physical 8 GB GPU remains untested.

```bash
git clone https://github.com/ujjwal-basnet/7-8gb-vram-video-generation.git
cd 7-8gb-vram-video-generation
uv sync --locked
uv run python download_models.py
uv run python monitor_run.py --frames 39 --output output/scene.mp4
```

Eight-step Turbo at the final resolution:

```bash
uv run python download_models.py --steps 8
uv run python monitor_run.py --mode native --steps 8 --frames 39 --seed 9175 --output output/dog-8step.mp4
```

Eight-step Turbo + SelfLift:

```bash
uv run python download_models.py --steps 8
uv run python monitor_run.py --steps 8 --frames 39 --output output/scene-8step.mp4
```

Four-step SelfLift is the default. Native mode runs every step at 640×384. SelfLift starts at 512×320 and switches to 640×384 after three of four steps or six of eight steps. Both modes export 24-fps video with audio. SelfLift does not guarantee identity, anatomy or motion accuracy. Try 39 frames first; 124 frames gives about five seconds. Edit the prompt or pass `--prompt-file path.txt`.

## Optional learned H3 upscaler

```bash
uv run python latent_upscaler.py
uv run python monitor_run.py --steps 8 --frames 39 --upscaler learned3d --output output/learned-8step.mp4
```

This adds the released 3D Conv v1 model (~659 MiB download) to the SelfLift transition, replacing the direct interpolated lift. H3 is unloaded while the upscaler runs; the paired VAE anchor and remaining diffusion steps are retained. The output stays 640×384. The completed eight-step learned-lift test produced visible stripe artifacts, so use `--upscaler interpolate` for SelfLift. The learned option remains experimental and is not the tutorial's Ultimate Upscale face-fix pipeline.

## Notebook

[View the input/output notebook](notebooks/video_generation.ipynb) · [Open in Google Colab](https://colab.research.google.com/github/ujjwal-basnet/7-8gb-vram-video-generation/blob/main/notebooks/video_generation.ipynb)

The notebook shows an editable dog-walking prompt, four/eight-step settings, native/SelfLift modes, model download, generation and a video player. It displays the new dog sample and the preserved reference GIFs. Select a GPU runtime on Colab; for local Jupyter, set `PROJECT` to your cloned repository directory.

Each run generates one clip. A continuous 15-second dog video under 8 GB has not been verified. The existing 15-second GIF is a historical reference sample, not a new dog result.

## Generated GIF previews

These loop directly on GitHub. GIF resolution/frame rate is reduced from the original 640×384, 24-fps videos. GIFs have no audio.

Dog walking: eight Turbo steps, native mode, 39 frames (1.625 seconds), seed 9175. The original video took 14m28s to generate on the T4.

![Eight-step dog walking](samples/dog-walking-8step.gif)

Same dog prompt with eight-step Turbo and interpolation SelfLift: 39 frames, seed 9175, 16m25s generation on the T4. Sampled GPU peak was 4.48 GiB; process RAM peaked at 8.56 GiB. Native was faster for this tested size and memory profile.

![Eight-step SelfLift dog walking](samples/dog-walking-selflift-8step.gif)

The following reference samples used four-step SelfLift:

Short scene:

![Short generated scene](samples/short-scene.gif)

15-second film:

![Three-scene generated film](samples/storm-guardian.gif)

Upstream models and code: [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [hybrid checkpoint](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [NF4 components](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4), [LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [SelfLift research](https://arxiv.org/abs/2609.02036), [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen). Their licenses/model terms apply. This is an experimental SelfLift-zero adaptation; no trained SelfLift LoRA or model weights are included.

The optional [H3 latent upscaler weights](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) use Apache-2.0 terms. Its network code is extracted from [LBH-123-AI's implementation](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) under the included [MIT notice](licenses/h3-latent-upscaler-MIT.txt).
