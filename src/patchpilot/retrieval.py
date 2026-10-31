"""Code retrieval via plain httpx REST against Qdrant, backed by Embedder protocol."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

import httpx

from patchpilot.providers.base import ProviderError

logger = logging.getLogger("patchpilot.retrieval")


@runtime_checkable
class Embedder(Protocol):
    """Embedder protocol for generating vector representations."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return float embeddings for each text in `texts`."""
        ...


class HashEmbedder:
    """Deterministic hashed-token embedder for tests, CI, and offline runs."""

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _embed_single(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.split()
        if not tokens:
            return vec

        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            val = int.from_bytes(digest[:8], byteorder="big", signed=False)
            idx = val % self.dim
            sign = 1.0 if (val >> 32) % 2 == 0 else -1.0
            vec[idx] += sign

        norm_sq = sum(x * x for x in vec)
        if norm_sq > 0.0:
            inv = 1.0 / (norm_sq**0.5)
            vec = [round(x * inv, 6) for x in vec]
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(t) for t in texts]


class OllamaEmbedder:
    """Embedder using host Ollama's POST /api/embed endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/api/embed"
        payload: dict[str, object] = {
            "model": self.model,
            "input": texts,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.post(url, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise ProviderError(f"ollama_embedder: transport error: {exc}") from exc

        if res.status_code != 200:
            raise ProviderError(
                f"ollama_embedder: HTTP {res.status_code} from /api/embed: {res.text[:200]}"
            )

        try:
            data = res.json()
        except ValueError as exc:
            raise ProviderError("ollama_embedder: invalid JSON response") from exc

        raw_embeddings = data.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise ProviderError("ollama_embedder: missing 'embeddings' list in response")

        result: list[list[float]] = []
        for item in raw_embeddings:
            if not isinstance(item, list):
                raise ProviderError("ollama_embedder: malformed embedding vector")
            result.append([float(x) for x in item])
        return result


class QdrantStore:
    """Plain httpx REST client for Qdrant with graceful degradation when down."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6333",
        timeout_s: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport

    async def is_healthy(self) -> bool:
        """Check whether Qdrant /healthz responds 200."""
        url = f"{self.base_url}/healthz"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.get(url)
                return res.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            logger.debug("Qdrant is down or unreachable: %s", exc)
            return False

    async def ensure_collection(self, collection_name: str, dim: int = 64) -> bool:
        """Ensure a collection exists with the specified vector dimension."""
        url = f"{self.base_url}/collections/{collection_name}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                check = await client.get(url)
                if check.status_code == 200:
                    return True
                payload = {
                    "vectors": {
                        "size": dim,
                        "distance": "Cosine",
                    }
                }
                res = await client.put(url, json=payload)
                return res.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Failed to ensure Qdrant collection %r: %s", collection_name, exc)
            return False

    async def upsert(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> bool:
        """Upsert points into collection. Returns False if down or failed."""
        if not points:
            return True
        url = f"{self.base_url}/collections/{collection_name}/points?wait=true"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.put(url, json={"points": points})
                return res.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Failed to upsert points into %r: %s", collection_name, exc)
            return False

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Search collection for nearest points. Returns [] if down or empty."""
        url = f"{self.base_url}/collections/{collection_name}/points/search"
        payload = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    return []
                data = res.json()
                results = data.get("result")
                if isinstance(results, list):
                    return [r for r in results if isinstance(r, dict)]
                return []
        except (httpx.HTTPError, OSError) as exc:
            logger.info("Qdrant search degraded gracefully for %r: %s", collection_name, exc)
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection if it exists."""
        url = f"{self.base_url}/collections/{collection_name}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                res = await client.delete(url)
                return res.status_code in (200, 404)
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Failed to delete Qdrant collection %r: %s", collection_name, exc)
            return False
