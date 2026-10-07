"""Keep H3 arithmetic in FP32 while using bounded FP16 attention on a T4."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def enable_memory_efficient_attention():
    from diffsynth.models import minimax_h3_dit
    def attention(q, k, v, scale=None, out=None):
        dtype=q.dtype
        # Attention is linear in V: scale it before casting and restore the
        # scale in FP32, preserving large finite values without FP16 overflow.
        vscale=(v.abs().amax()/8192).clamp_min(1)
        values=[q.to(torch.float16),k.to(torch.float16),(v/vscale).to(torch.float16)]
        if not all(bool(torch.isfinite(tensor).all()) for tensor in values):
            raise ValueError("Attention values exceed finite FP16 range")
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output=F.scaled_dot_product_attention(*values,scale=scale)
        if out is not None:
            for start in range(0, q.shape[-2], 2048):
                section=(Ellipsis, slice(start, start+2048), slice(None))
                out[section].copy_(output[section])
                out[section].mul_(vscale)
            return out
        return (output.float()*vscale).to(dtype)
    minimax_h3_dit.attention_forward=attention
