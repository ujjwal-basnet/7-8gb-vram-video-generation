# 7–8 GB VRAM Video Generation

Python/PyTorch video generation with H3 hybrid INT8, Turbo and experimental SelfLift-zero. No ComfyUI runtime. The default prompt is one dog walking with a steady camera and simple background. `prompts/lantern.txt` is another simple prompt.

Requires Linux, Python 3.12, a CUDA GPU, `ffmpeg`, [uv](https://docs.astral.sh/uv/getting-started/installation/), around 60 GB disk and preferably 16 GB system RAM. Reference four-step runs on a Colab T4 peaked at 6.44 GiB sampled GPU memory and 9.41 GiB process RAM for a three-scene film. A physical 8 GB GPU and the eight-step low-memory option remain untested.

```bash
git clone https://github.com/ujjwal-basnet/7-8gb-vram-video-generation.git
cd 7-8gb-vram-video-generation
uv sync --locked
uv run python download_models.py
uv run python monitor_run.py --frames 39 --output output/scene.mp4
```

Eight-step Turbo + SelfLift:

```bash
uv run python download_models.py --steps 8
uv run python monitor_run.py --steps 8 --frames 39 --output output/scene-8step.mp4
```

Four-step is the measured default. Eight-step switches to high resolution after six steps. Both start at 512×320 and finish at 640×384, 24 fps with audio. SelfLift refines at the higher resolution during denoising; it does not guarantee identity, anatomy or motion accuracy. Try 39 frames first; 124 frames gives about five seconds. Edit the prompt or pass `--prompt-file path.txt`.

For a 15-second dog-walking film, run `uv run python render_story.py --steps 4` (or `--steps 8` after downloading that adapter). This uses the three prompts in `prompts/dog-walking-scenes.json`, with one dog walking in each shot. Scenes run one at a time and are joined and trimmed to 15 seconds in `output/dog-walking/film-15s.mp4`. To render the earlier fantasy story, pass `--scenes prompts/scenes.json --output-dir output/storm-guardian`. The reference four-step film took about 54 minutes on T4, excluding setup/downloads. Eight-step time and memory are not measured yet.

## Generated GIF previews

These loop directly on GitHub. They preview previously completed four-step runs with the same low-memory recipe; they are not dog/lantern results or an eight-step benchmark. GIF resolution/frame rate is reduced from the original 640×384, 24 fps videos. GIFs have no audio.

Short scene:

![Short generated scene](samples/short-scene.gif)

15-second film:

![Three-scene generated film](samples/storm-guardian.gif)

Upstream models and code: [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), [hybrid checkpoint](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [NF4 components](https://huggingface.co/DiffSynth-Studio/MiniMax-H3-NF4), [LightX2V Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), [SelfLift research](https://arxiv.org/abs/2609.02036), [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen). Their licenses/model terms apply. This is an experimental SelfLift-zero adaptation; no trained SelfLift LoRA or model weights are included.
