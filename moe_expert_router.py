"""Fine-grained MoE expert router (reference implementation).

Simulates top-k sparse gating with auxiliary-loss-free load balancing (per-expert
bias adjustment), the routing style used by modern fine-grained MoE models.
The expert counts used here are an illustrative reference configuration, not
official Yi-Lightning specifications (01.AI has not published them).
"""

import math
import random
from typing import Dict, List


class AuxLossFreeRouter:
    """Top-k router that balances load by nudging per-expert bias terms."""

    def __init__(self, num_experts: int = 64, top_k: int = 8, bias_update_rate: float = 0.01, seed: int = 7):
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        self.bias = [0.0] * num_experts
        self.rng = random.Random(seed)

    def _affinity(self) -> List[float]:
        # Skewed synthetic affinities: low-index experts are naturally "popular".
        return [self.rng.gauss(1.0 - i / self.num_experts, 0.3) for i in range(self.num_experts)]

    def route_token(self) -> List[int]:
        scores = self._affinity()
        biased = [s + b for s, b in zip(scores, self.bias)]
        return sorted(range(self.num_experts), key=lambda i: biased[i], reverse=True)[: self.top_k]

    def route_batch(self, num_tokens: int, balance: bool = True) -> Dict[str, float]:
        load = [0] * self.num_experts
        for _ in range(num_tokens):
            for expert in self.route_token():
                load[expert] += 1
        if balance:
            mean = sum(load) / self.num_experts
            # Bias only affects selection, never the gate weights -> no aux loss term.
            for i, l in enumerate(load):
                self.bias[i] += self.bias_update_rate * (1 if l < mean else -1)
        mean = sum(load) / self.num_experts
        std = math.sqrt(sum((l - mean) ** 2 for l in load) / self.num_experts)
        return {"max_load": max(load), "min_load": min(load), "cv": round(std / mean, 4)}


def main() -> None:
    router = AuxLossFreeRouter()
    first = router.route_batch(2048)
    for _ in range(60):
        last = router.route_batch(2048)
    print("Initial load stats:", first)
    print("Balanced load stats:", last)
    assert last["cv"] < first["cv"], "bias balancing should reduce load imbalance"
    print("Router validation passed.")


if __name__ == "__main__":
    main()
