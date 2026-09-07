from __future__ import annotations

import torch


def split_fused_qkv(output: torch.Tensor, *, num_attention_heads: int, kv_channels: int, multi_query_group_num: int):
    q_size = int(num_attention_heads) * int(kv_channels)
    kv_size = int(multi_query_group_num) * int(kv_channels)
    expected = q_size + 2 * kv_size
    if output.shape[-1] != expected:
        raise ValueError(f"expected fused QKV width {expected}, got {output.shape[-1]}")
    q = output[..., :q_size]
    k = output[..., q_size:q_size + kv_size]
    v = output[..., q_size + kv_size:q_size + 2 * kv_size]
    return q, k, v


def apply_kv_delta(output: torch.Tensor, *, key_delta: torch.Tensor, value_delta: torch.Tensor,
                   num_attention_heads: int, kv_channels: int, multi_query_group_num: int, scale: float = 1.0):
    q, k, v = split_fused_qkv(
        output,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        multi_query_group_num=multi_query_group_num,
    )
    key_delta = key_delta.to(device=output.device, dtype=output.dtype)
    value_delta = value_delta.to(device=output.device, dtype=output.dtype)
    if key_delta.numel() != k.shape[-1] or value_delta.numel() != v.shape[-1]:
        raise ValueError("delta sizes must match fused K/V widths")
    return torch.cat((q, k + float(scale) * key_delta, v + float(scale) * value_delta), dim=-1)


class FusedQKVRecorder:
    """Record pooled K/V projection outputs from ChatGLM-style fused query_key_value modules."""
    def __init__(self, model, *, layer_indices, num_attention_heads, kv_channels, multi_query_group_num):
        self.model = model
        self.layer_indices = [int(i) for i in layer_indices]
        self.num_attention_heads = int(num_attention_heads)
        self.kv_channels = int(kv_channels)
        self.multi_query_group_num = int(multi_query_group_num)
        self.handles = []
        self.values = {}
        self._modules = [m for name, m in model.named_modules() if name.endswith("query_key_value")]
        if not self._modules:
            raise ValueError("no ChatGLM-style query_key_value modules found")
        bad = [i for i in self.layer_indices if i < 0 or i >= len(self._modules)]
        if bad:
            raise ValueError(f"layer indices out of range: {bad}; found {len(self._modules)} QKV modules")

    def _hook_for(self, layer_idx):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            _, k, v = split_fused_qkv(
                tensor,
                num_attention_heads=self.num_attention_heads,
                kv_channels=self.kv_channels,
                multi_query_group_num=self.multi_query_group_num,
            )
            k_pool = k.reshape(-1, k.shape[-1]).float().mean(dim=0)
            v_pool = v.reshape(-1, v.shape[-1]).float().mean(dim=0)
            self.values[layer_idx] = torch.cat((k_pool, v_pool)).detach().to("cpu", dtype=torch.float16)
        return hook

    def clear(self):
        self.values.clear()

    def __enter__(self):
        self.clear()
        for idx in self.layer_indices:
            self.handles.append(self._modules[idx].register_forward_hook(self._hook_for(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def vector(self):
        missing = [i for i in self.layer_indices if i not in self.values]
        if missing:
            raise RuntimeError(f"selected layers were not observed during forward: {missing}")
        return torch.cat([self.values[i] for i in self.layer_indices], dim=0)


class FusedQKVSteerer:
    """Apply a concatenated [K,V] direction to selected fused-QKV layers."""
    def __init__(self, model, *, layer_indices, direction, num_attention_heads, kv_channels,
                 multi_query_group_num, scale=1.0):
        self.model = model
        self.layer_indices = [int(i) for i in layer_indices]
        self.num_attention_heads = int(num_attention_heads)
        self.kv_channels = int(kv_channels)
        self.multi_query_group_num = int(multi_query_group_num)
        self.scale = float(scale)
        self._modules = [m for name, m in model.named_modules() if name.endswith("query_key_value")]
        if not self._modules:
            raise ValueError("no ChatGLM-style query_key_value modules found")
        bad = [i for i in self.layer_indices if i < 0 or i >= len(self._modules)]
        if bad:
            raise ValueError(f"layer indices out of range: {bad}; found {len(self._modules)} QKV modules")
        kv_size = self.multi_query_group_num * self.kv_channels
        per_layer = 2 * kv_size
        direction = torch.as_tensor(direction).flatten()
        expected = len(self.layer_indices) * per_layer
        if direction.numel() != expected:
            raise ValueError(f"direction length {direction.numel()} != expected {expected}")
        self.layer_deltas = {}
        for pos, idx in enumerate(self.layer_indices):
            chunk = direction[pos * per_layer:(pos + 1) * per_layer]
            self.layer_deltas[idx] = (chunk[:kv_size], chunk[kv_size:])
        self.handles = []

    def _hook_for(self, layer_idx):
        key_delta, value_delta = self.layer_deltas[layer_idx]
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            steered = apply_kv_delta(
                tensor,
                key_delta=key_delta,
                value_delta=value_delta,
                num_attention_heads=self.num_attention_heads,
                kv_channels=self.kv_channels,
                multi_query_group_num=self.multi_query_group_num,
                scale=self.scale,
            )
            if isinstance(output, tuple):
                return (steered, *output[1:])
            if isinstance(output, list):
                return [steered, *output[1:]]
            return steered
        return hook

    def __enter__(self):
        for idx in self.layer_indices:
            self.handles.append(self._modules[idx].register_forward_hook(self._hook_for(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
