"""Render three five-second scenes, then join and trim to fifteen seconds."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from settings import ROOT


def render_story(steps=4, scenes_file=ROOT / "prompts/dog-walking-scenes.json", output_dir=ROOT / "output/dog-walking"):
    from imageio_ffmpeg import get_ffmpeg_exe
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(Path(scenes_file).read_text())
    clips = []
    for scene in scenes:
        clip = output / f"scene-{scene['scene']}.mp4"
        subprocess.run([sys.executable, str(ROOT / "monitor_run.py"),
                        "--prompt-file", str(ROOT / "prompts" / scene["prompt_file"]),
                        "--steps", str(steps), "--frames", "124",
                        "--seed", str(scene["seed"]), "--output", str(clip)], check=True)
        clips.append(clip)
    listing = output / "concat.txt"
    listing.write_text("".join(f"file '{clip.name}'\n" for clip in clips))
    final = output / "film-15s.mp4"
    subprocess.run([get_ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-t", "15", "-c:v", "libx264", "-crf", "18",
                    "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(final)], check=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--scenes", type=Path, default=ROOT / "prompts/dog-walking-scenes.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/dog-walking")
    args = parser.parse_args()
    print(render_story(args.steps, args.scenes, args.output_dir))
