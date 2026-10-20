"""Model registry + explicit provider routing for paid runs.

ADDITIVE. The legacy routing inside `pge.agents.llama_together` (model id in
`_OPENROUTER_MODELS` -> OpenRouter, else Together) is untouched, so every
committed script keeps its exact behavior. New scripts (eval/paid/*) resolve
a model through this registry, which names the provider explicitly, so any
model string can be routed to either provider.

Prices are USD per million tokens (input, output). Entries marked
VERIFY are best-known list prices, NOT checked against the provider's
current pricing page: verify on the provider site before a large run. Cost
caps use these prices (and, where the provider returns one, the
provider-reported per-call cost, whichever is larger).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pge.agents.llama_together import (
    MODEL_LLAMA_8B,
    MODEL_LLAMA_70B,
    OPENROUTER_URL,
    PRICE_PER_M,
    TOGETHER_URL,
)

VERIFY = "verify on provider site"


@dataclass(frozen=True)
class ProviderInfo:
    url: str
    key_env: str


PROVIDERS: dict[str, ProviderInfo] = {
    "together": ProviderInfo(TOGETHER_URL, "TOGETHER_API_KEY"),
    "openrouter": ProviderInfo(OPENROUTER_URL, "OPENROUTER_API_KEY"),
}


@dataclass(frozen=True)
class ModelSpec:
    short_name: str  # used in output file names: <short_name>_<seed>.jsonl
    model_id: str  # provider model slug
    provider: str  # key of PROVIDERS
    price_in: float  # $/M input tokens
    price_out: float  # $/M output tokens
    price_status: str  # provenance of the price

    @property
    def url(self) -> str:
        return PROVIDERS[self.provider].url

    @property
    def key_env(self) -> str:
        return PROVIDERS[self.provider].key_env

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in * self.price_in / 1e6 + tokens_out * self.price_out / 1e6


_SPECS = [
    # Existing tiers: same ids, providers and prices as llama_together.
    ModelSpec("70B", MODEL_LLAMA_70B, "together", *PRICE_PER_M[MODEL_LLAMA_70B],
              "ledger (2026-09 runs)"),
    ModelSpec("8B", MODEL_LLAMA_8B, "openrouter", *PRICE_PER_M[MODEL_LLAMA_8B],
              "conservative upper bound (ledger)"),
    # New families (prices: VERIFY).
    ModelSpec("qwen72b-together", "Qwen/Qwen2.5-72B-Instruct-Turbo", "together",
              1.20, 1.20, VERIFY),
    ModelSpec("qwen72b", "qwen/qwen-2.5-72b-instruct", "openrouter", 0.40, 0.40, VERIFY),
    ModelSpec("mistral-small-24b", "mistralai/mistral-small-3.1-24b-instruct", "openrouter",
              0.35, 0.60, VERIFY),
    ModelSpec("gpt-4o-mini", "openai/gpt-4o-mini", "openrouter", 0.15, 0.60, VERIFY),
]

REGISTRY: dict[str, ModelSpec] = {s.short_name: s for s in _SPECS}

# Cheap model for the LLM-simulated patient and the feedback extractor.
PATIENT_DEFAULT = "8B"
EXTRACTOR_DEFAULT = "8B"


def resolve(
    name: str,
    provider: Optional[str] = None,
    short_name: Optional[str] = None,
    price_in: Optional[float] = None,
    price_out: Optional[float] = None,
) -> ModelSpec:
    """Registry short name, or a registered model id, or (with an explicit
    provider and prices) an arbitrary model string."""
    if name in REGISTRY and provider in (None, REGISTRY[name].provider):
        spec = REGISTRY[name]
    else:
        hits = [s for s in _SPECS if s.model_id == name and provider in (None, s.provider)]
        if hits:
            spec = hits[0]
        else:
            if provider is None:
                raise SystemExit(
                    f"model '{name}' is not in the registry; pass --provider "
                    f"({'/'.join(PROVIDERS)}) and --price-in/--price-out"
                )
            if provider not in PROVIDERS:
                raise SystemExit(f"unknown provider '{provider}' (known: {sorted(PROVIDERS)})")
            if price_in is None or price_out is None:
                raise SystemExit(
                    f"unregistered model '{name}': pass --price-in and --price-out "
                    "($/M tokens) so the cost cap can be enforced"
                )
            spec = ModelSpec(
                short_name or name.split("/")[-1].lower(), name, provider,
                price_in, price_out, "user-supplied",
            )
    if short_name or price_in is not None or price_out is not None:
        spec = ModelSpec(
            short_name or spec.short_name,
            spec.model_id,
            spec.provider,
            spec.price_in if price_in is None else price_in,
            spec.price_out if price_out is None else price_out,
            spec.price_status if price_in is None and price_out is None else "user-supplied",
        )
    return spec
