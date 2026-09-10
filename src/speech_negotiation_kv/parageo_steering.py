from __future__ import annotations

import torch

from .kv_hooks import apply_kv_delta


class ScheduledFusedQKVSteerer:
    """Apply a per-generation-step sequence of concatenated [K,V] directions.

    Each row in ``directions`` is the concatenation of [K,V] deltas for all
    selected layers. The first selected QKV layer advances the schedule once per
    model forward; all other selected layers reuse that step's direction.
    """

    def __init__(self, model, *, layer_indices, directions, num_attention_heads: int,
                 kv_channels: int, multi_query_group_num: int, scale: float = 1.0):
        self.model = model
        self.layer_indices = [int(value) for value in layer_indices]
        if not self.layer_indices:
            raise ValueError("layer_indices must be non-empty")
        self.num_attention_heads = int(num_attention_heads)
        self.kv_channels = int(kv_channels)
        self.multi_query_group_num = int(multi_query_group_num)
        self.scale = float(scale)
        self._modules = [m for name, m in model.named_modules() if name.endswith("query_key_value")]
        if not self._modules:
            raise ValueError("no ChatGLM-style query_key_value modules found")
        bad = [idx for idx in self.layer_indices if idx < 0 or idx >= len(self._modules)]
        if bad:
            raise ValueError(f"layer indices out of range: {bad}")
        kv_size = self.multi_query_group_num * self.kv_channels
        self.per_layer = 2 * kv_size
        tensor = torch.as_tensor(directions, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        expected = len(self.layer_indices) * self.per_layer
        if tensor.ndim != 2 or tensor.shape[1] != expected or tensor.shape[0] < 1:
            raise ValueError(f"directions must be [steps, {expected}]")
        self.directions = tensor
        self.handles = []
        self.step_index = -1
        self.current_direction = tensor[0]

    def _layer_delta(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.layer_indices.index(layer_idx)
        chunk = self.current_direction[position * self.per_layer:(position + 1) * self.per_layer]
        kv_size = self.per_layer // 2
        return chunk[:kv_size], chunk[kv_size:]

    def _hook_for(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if layer_idx == self.layer_indices[0]:
                self.step_index += 1
                schedule_index = min(self.step_index, self.directions.shape[0] - 1)
                self.current_direction = self.directions[schedule_index]
            key_delta, value_delta = self._layer_delta(layer_idx)
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
        self.step_index = -1
        self.current_direction = self.directions[0]
        for idx in self.layer_indices:
            self.handles.append(self._modules[idx].register_forward_hook(self._hook_for(idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
