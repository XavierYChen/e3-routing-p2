"""Read routing tensors with removable forward hooks; model forward remains untouched."""

from __future__ import annotations

from typing import Any, Self

import numpy as np


class SpatialRoutingCollector:
    def __init__(self, model: Any, family: str) -> None:
        self.model, self.family = model, family.lower()
        self.handles: list[Any] = []
        self.captures: list[dict[str, Any]] = []

    def __enter__(self) -> Self:
        wanted = {"moe": "EfficientSpatialRouter", "mot": "_MoTRouter", "latent": "LatentRouter"}[self.family]
        for name, module in self.model.named_modules():
            if type(module).__name__ == wanted:
                self.handles.append(module.register_forward_hook(self._hook(name, module)))
        if not self.handles:
            raise ValueError(f"unsupported: no {wanted} found")
        return self

    def _hook(self, name: str, module: Any):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            import torch

            if self.family == "mot":
                probs, indices = output[0], output[1]
                granularity, source = "token", "router_softmax[B,E,H,W]"
            elif self.family == "moe":
                weights, indices = output[0], output[1]
                experts = int(getattr(module, "num_experts", weights.shape[-1]))
                probs = weights.new_zeros((weights.shape[0], experts))
                probs.scatter_add_(1, indices.long(), weights)
                probs = probs[:, :, None, None]
                granularity = "image"
                source = "router_topk[B,K]"
            else:
                probs = output[1][:, :, None, None]
                indices = probs.squeeze(-1).squeeze(-1).argsort(dim=1, descending=True)
                granularity, source = "image", "router_softmax[B,E]"
            probs = probs.detach().float().cpu()
            if not torch.isfinite(probs).all() or (probs < 0).any():
                raise ValueError(f"{name}: invalid routing probabilities")
            error = float((probs.sum(1) - 1).abs().max())
            if error > 1e-4:
                raise ValueError(f"{name}: probabilities do not sum to one ({error})")
            self.captures.append({
                "layer": name,
                "maps": probs[0].numpy().astype(np.float32),
                "indices": indices.detach().cpu().numpy().tolist(),
                "num_experts": int(probs.shape[1]),
                "top_k": int(indices.shape[1]),
                "spatial_granularity": granularity,
                "source": source,
                "probability_sum_max_error": error,
            })
        return capture

    def representative(self) -> dict[str, Any]:
        if not self.captures:
            raise ValueError("no routing tensors captured")
        return max(self.captures, key=lambda item: item["maps"].shape[-2] * item["maps"].shape[-1])

    def __exit__(self, *_args: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
