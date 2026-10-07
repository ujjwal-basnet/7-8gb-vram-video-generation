"""Fetch the pinned hybrid, companion components and selected Turbo adapter."""
from pathlib import Path
from hybrid_checkpoint import download as download_hybrid
from settings import ROOT

NF4_REV = "363fdc8fbd7ae55f5b9e7fb87cf3d508df28ee0d"
H3_REV = "42ed227ee7df40d41602854ae760620d6eb651fe"
TURBO_REV = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
TURBO_FILES = {steps: f"minimax_h3_fl2v_turbo_{steps}step_v1.0_768p_bf16.safetensors"
               for steps in (4, 8)}
TURBO = TURBO_FILES[4]
COMPANIONS = {
    "minimax-h3-text-encoder-nf4.safetensors": 15324775807,
    "video_vae_nf4.safetensors": 1613201536,
    "audio_vae_nf4.safetensors": 284004112,
}


def download_models(steps=4):
    from huggingface_hub import hf_hub_download, snapshot_download
    download_hybrid()
    for filename, size in COMPANIONS.items():
        path = Path(hf_hub_download("DiffSynth-Studio/MiniMax-H3-NF4", filename,
                    revision=NF4_REV, local_dir=ROOT / "models/nf4"))
        if path.stat().st_size != size:
            raise ValueError(f"Wrong size for {filename}")
    snapshot_download("MiniMaxAI/MiniMax-H3", revision=H3_REV,
                      allow_patterns=["FL2VA/processor/*"], local_dir=ROOT / "models/h3")
    hf_hub_download("lightx2v/Minimax-h3-Turbo", TURBO_FILES[steps], revision=TURBO_REV,
                   local_dir=ROOT / "models/turbo")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=[4, 8], default=4)
    download_models(parser.parse_args().steps)
