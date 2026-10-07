"""Bound H3 inference temporaries without splitting the attention sequence.

Adapted from DiffSynth-Studio's streamed layers and H3 MLP/attention.
Changes batch pointwise work and release fused QKV storage before attention.
Apache-2.0 terms: licenses/diffsynth-APACHE-2.0.txt.
"""
import torch
import torch.nn.functional as F

ROW_LIMIT = 4096
CHUNK_ROWS = 2048


def row_chunks(operation, x, out_features):
    rows = x.reshape(-1, x.shape[-1])
    if len(rows) <= ROW_LIMIT:
        return operation(x)
    output = x.new_empty(len(rows), out_features)
    for start in range(0, len(rows), CHUNK_ROWS):
        batch = rows[start:start + CHUNK_ROWS]
        output[start:start + len(batch)].copy_(operation(batch))
    return output.reshape(*x.shape[:-1], out_features)


def materialize_linear(layer):
    """Load each streamed weight once, then reuse it for all row batches."""
    from diffsynth.core.vram.layers import AutoWrappedLinear, AutoWrappedQuantizedModule
    if not isinstance(layer, (AutoWrappedLinear, AutoWrappedQuantizedModule)):
        return layer
    if layer.state == 1 and (layer.vram_limit is None or layer.check_free_vram()):
        layer.preparing()
    if isinstance(layer, AutoWrappedQuantizedModule):
        operation = layer.computation_module()
    else:
        weight, bias = layer.computation()
        operation = lambda rows: layer.linear_forward(rows, weight, bias)

    def compute(rows):
        output = operation(rows)
        return layer.lora_forward(rows, output) if layer.lora_A_weights else output
    return compute


def enable_bounded_rows():
    """Apply to the pinned DiffSynth classes before building the pipeline."""
    from diffsynth.core.vram.layers import AutoWrappedLinear, AutoWrappedQuantizedModule
    from diffsynth.models import minimax_h3_dit as dit, minimax_h3_dit_comfy as comfy

    def linear(self, x):
        return row_chunks(materialize_linear(self), x, self.out_features)

    original_mlp = dit.MiniMaxH3MLP.forward

    def mlp(self, x):
        if x.shape[0] <= ROW_LIMIT:
            return original_mlp(self, x)
        first, second = materialize_linear(self.fc1), materialize_linear(self.fc2)

        def compute(rows):
            gate, up = first(rows).chunk(2, dim=-1)
            return second(F.silu(gate) * up)
        return row_chunks(compute, x, self.fc2.out_features)

    original_attention = comfy._comfy_attention_forward

    def attention(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None):
        if x.shape[0] <= ROW_LIMIT:
            return original_attention(self, x, rope_freqs=rope_freqs,
                                      cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        qkv = self.qkv_proj(x).view(len(x), 3, self.num_heads, self.head_dim)
        q, k = self.q_norm(qkv[:, 0]), self.k_norm(qkv[:, 1])
        v = qkv[:, 2].contiguous()
        del qkv
        if rope_freqs is not None:
            for start in range(0, len(x), CHUNK_ROWS):
                section = slice(start, start + CHUNK_ROWS)
                q[section].copy_(dit._apply_rope(q[section], rope_freqs[section]))
                k[section].copy_(dit._apply_rope(k[section], rope_freqs[section]))
        output = torch.empty_like(q)
        bounds = cu_seqlens.tolist()
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            packed = [item[start:stop].transpose(0, 1).unsqueeze(0) for item in (q, k, v)]
            target = output[start:stop].transpose(0, 1).unsqueeze(0)
            dit.attention_forward(*packed, scale=self.softmax_scale, out=target)
        del q, k, v, packed
        return self.out_proj(output.reshape(len(x), -1))

    AutoWrappedLinear.forward = linear
    AutoWrappedQuantizedModule.forward = linear
    dit.MiniMaxH3MLP.forward = mlp
    comfy._comfy_attention_forward = attention
