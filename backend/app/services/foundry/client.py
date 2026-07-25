"""Thin wrapper around the Foundry Local SDK.

Two things this module exists to hide from the rest of the app:

1. ``FoundryLocalManager`` is a process-wide singleton that may only be
   initialised once, which does not fit FastAPI's threaded request model
   without a guard.
2. Model *variants* must sometimes be pinned to CPU: on machines where the
   OpenVINO GPU execution provider fails to initialise, the default (GPU)
   variant raises ``Could not find an implementation for EPContext`` at load
   time. Selecting the CPU variant avoids that entirely.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_models: "LocalModels | None" = None


def _ensure_manager():
    """Initialise the SDK singleton against the shared model cache."""
    from foundry_local_sdk import Configuration, FoundryLocalManager

    if FoundryLocalManager.instance is None:
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        FoundryLocalManager.initialize(
            Configuration(
                app_name="reglens",
                model_cache_dir=str(settings.model_cache_dir),
            )
        )
    return FoundryLocalManager.instance


def _select_cpu_variant(model: Any) -> None:
    """Pin the model to its CPU variant when one exists."""
    if not settings.force_cpu_variant:
        return
    for variant in model.variants:
        runtime = getattr(variant.info, "runtime", None)
        if runtime is not None and str(runtime.device_type) == "CPU":
            model.select_variant(variant)
            return


class LocalModels:
    """Holds the loaded chat + embedding models and exposes a small API."""

    def __init__(self) -> None:
        # The SDK is imported lazily (inside _ensure_manager) so that importing
        # this module in tests that stub the models out does not require it.
        manager = _ensure_manager()

        logger.info("Loading embedding model %s", settings.embedding_model_alias)
        self.embedding_model = manager.catalog.get_model(settings.embedding_model_alias)
        if self.embedding_model is None:
            raise RuntimeError(
                f"Embedding model '{settings.embedding_model_alias}' is not in the "
                "Foundry Local catalog. Run `foundry model list` to see what is available."
            )
        _select_cpu_variant(self.embedding_model)
        self.embedding_model.download()
        self.embedding_model.load()
        self._embedding_client = self.embedding_model.get_embedding_client()

        logger.info("Loading chat model %s", settings.chat_model_alias)
        self.chat_model = manager.catalog.get_model(settings.chat_model_alias)
        if self.chat_model is None:
            raise RuntimeError(
                f"Chat model '{settings.chat_model_alias}' is not in the Foundry Local "
                "catalog. Run `foundry model list` to see what is available."
            )
        _select_cpu_variant(self.chat_model)
        self.chat_model.download()
        self.chat_model.load()
        self._chat_client = self.chat_model.get_chat_client()

    # --- embeddings -----------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        response = self._embedding_client.generate_embedding(text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._embedding_client.generate_embeddings(texts)
        return [item.embedding for item in response.data]

    # --- chat -----------------------------------------------------------------

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        self._chat_client.settings.max_tokens = max_tokens or settings.max_answer_tokens
        self._chat_client.settings.temperature = 0.2
        completion = self._chat_client.complete_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return (completion.choices[0].message.content or "").strip()

    def chat_stream(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None):
        """Yield answer fragments as the model produces them."""
        self._chat_client.settings.max_tokens = max_tokens or settings.max_answer_tokens
        self._chat_client.settings.temperature = 0.2
        for chunk in self._chat_client.complete_streaming_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        ):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    @property
    def chat_model_id(self) -> str:
        return self.chat_model.id

    @property
    def embedding_model_id(self) -> str:
        return self.embedding_model.id


def get_models() -> LocalModels:
    """Return the shared ``LocalModels``, loading them on first use."""
    global _models
    if _models is None:
        with _lock:
            if _models is None:
                _models = LocalModels()
    return _models


def is_loaded() -> bool:
    return _models is not None


def probe() -> dict[str, Any]:
    """Report Foundry Local availability without forcing a model download."""
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager

        if FoundryLocalManager.instance is None:
            settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
            FoundryLocalManager.initialize(
                Configuration(
                    app_name="reglens",
                    model_cache_dir=str(settings.model_cache_dir),
                )
            )
        manager = FoundryLocalManager.instance
        chat = manager.catalog.get_model(settings.chat_model_alias)
        embed = manager.catalog.get_model(settings.embedding_model_alias)
        return {
            "available": True,
            "models_loaded": is_loaded(),
            "chat_model": settings.chat_model_alias,
            "chat_model_in_catalog": chat is not None,
            "embedding_model": settings.embedding_model_alias,
            "embedding_model_in_catalog": embed is not None,
        }
    except Exception as exc:  # noqa: BLE001 - health endpoint must not raise
        return {
            "available": False,
            "models_loaded": False,
            "chat_model": settings.chat_model_alias,
            "embedding_model": settings.embedding_model_alias,
            "error": str(exc),
        }
