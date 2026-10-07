"""Generate one scene with the fixed low-VRAM recipe."""
import argparse
from pathlib import Path
from settings import ROOT, RenderSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "prompts/dog-walking.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "output/scene.mp4")
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=9175)
    parser.add_argument("--steps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--upscaler", choices=["interpolate", "learned3d"], default="interpolate")
    args = parser.parse_args()
    from pipeline import render
    print(render(args.prompt_file.read_text().strip(), args.output,
                 RenderSettings(frames=args.frames, seed=args.seed, steps=args.steps,
                                upscaler=args.upscaler)), flush=True)


if __name__ == "__main__":
    main()
