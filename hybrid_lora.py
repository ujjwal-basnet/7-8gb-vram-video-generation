"""Adapt LightX2V's native H3 QKV layout to the selected Comfy-format DiT."""
from diffsynth.utils.lora.minimax_h3 import MiniMaxH3LoRALoader, MiniMaxH3LoRAConverter


def turbo_metadata(path):
    """Read original adapter ranks before QKV fusion changes their size."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
        ranks = {weights.get_slice(key).get_shape()[0] for key in weights.keys()
                 if key.endswith(".lora_A.default.weight")}
    if len(ranks) != 1 or "alpha" not in metadata:
        raise ValueError("Expected pinned Turbo adapter alpha and a uniform rank")
    rank = ranks.pop()
    alpha = float(metadata["alpha"])
    return {"alpha": alpha, "rank": rank, "scale": alpha / rank}


class HybridTurboLoader(MiniMaxH3LoRALoader):
    def convert_state_dict(self, state_dict, suffix=".weight"):
        if not self.is_lightx2v_format(state_dict):
            raise ValueError("HybridTurboLoader expects the pinned LightX2V adapter format")
        aligned = MiniMaxH3LoRAConverter.align_to_diffsynth_format(state_dict)
        for key, value in aligned.items():
            if key.endswith(".qkv_proj.lora_B.default.weight"):
                heads = value.shape[0] // (3 * 128)
                aligned[key] = value.reshape(heads, 3, 128, -1).permute(1, 0, 2, 3).reshape_as(value)
        return super().convert_state_dict(aligned, suffix=suffix)
