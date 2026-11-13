"""Code retrieval via plain httpx REST against Qdrant, backed by Embedder protocol."""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from patchpilot.pipeline.normalization import is_modifiable_py_file
from patchpilot.providers.base import ProviderError

logger = logging.getLogger("patchpilot.retrieval")


def case_collection_name(files: Mapping[str, str]) -> str:
    """Derive a per-case Qdrant collection name from the case content hash."""
    hasher = hashlib.sha256()
    for path in sorted(files):
        hasher.update(path.replace("\\", "/").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(files[path].encode("utf-8"))
        hasher.update(b"\x00")
    return f"patchpilot_case_{hasher.hexdigest()[:16]}"


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
        tokens = re.findall(r"\w+", text.lower())
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
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderError(f"ollama_embedder: transport error: {exc}") from exc

        if res.status_code != 200:
            raise ProviderError(
                f"ollama_embedder: HTTP {res.status_code} from /api/embed: {res.text[:200]}"
            )

        try:
            data = res.json()
        except ValueError as exc:
            raise ProviderError("ollama_embedder: invalid JSON response") from exc

        if not isinstance(data, dict):
            raise ProviderError("ollama_embedder: JSON response is not an object")

        raw_embeddings = data.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise ProviderError("ollama_embedder: missing 'embeddings' list in response")

        result: list[list[float]] = []
        for item in raw_embeddings:
            if not isinstance(item, list) or not item:
                raise ProviderError("ollama_embedder: malformed embedding vector")
            try:
                vec = [float(x) for x in item if not isinstance(x, bool)]
            except (TypeError, ValueError) as exc:
                raise ProviderError("ollama_embedder: non-numeric embedding element") from exc
            if len(vec) != len(item) or any(math.isnan(v) or math.isinf(v) for v in vec):
                raise ProviderError("ollama_embedder: invalid float in embedding vector")
            result.append(vec)

        if len(result) != len(texts):
            raise ProviderError(
                f"ollama_embedder: expected {len(texts)} embeddings, got {len(result)}"
            )
        return result


def _node_line_span(node: ast.stmt) -> tuple[int, int]:
    """Return 1-based `(start_line, end_line)` including decorators for an AST statement."""
    decorators = getattr(node, "decorator_list", [])
    start_line = min([node.lineno] + [d.lineno for d in decorators])
    end_line = getattr(node, "end_lineno", None) or node.lineno
    return (start_line, end_line)


def _clean_chunk_path(file_path: str) -> str:
    norm = file_path.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def extract_ast_chunks(file_path: str, content: str) -> list[dict[str, Any]]:
    """Chunk a Python file by AST definitions (`ClassDef`, `FunctionDef`, `AsyncFunctionDef`).

    Returns chunk dicts with keys `{"file", "symbol", "start_line", "end_line", "text"}`.
    Falls back to bounded line windows on `SyntaxError`.
    """
    norm_path = _clean_chunk_path(file_path)
    lines = content.splitlines()
    if not lines or not content.strip():
        return []

    try:
        tree = ast.parse(content, filename=norm_path)
    except SyntaxError:
        chunks: list[dict[str, Any]] = []
        window = 60
        step = 50
        for idx in range(0, len(lines), step):
            end_idx = min(len(lines), idx + window)
            chunk_text = "\n".join(lines[idx:end_idx])
            if chunk_text.strip():
                chunks.append(
                    {
                        "file": norm_path,
                        "symbol": f"<lines:{idx + 1}-{end_idx}>",
                        "start_line": idx + 1,
                        "end_line": end_idx,
                        "text": chunk_text,
                    }
                )
            if end_idx >= len(lines):
                break
        return chunks

    top_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    if not top_defs:
        return [
            {
                "file": norm_path,
                "symbol": "<module>",
                "start_line": 1,
                "end_line": len(lines),
                "text": "\n".join(lines),
            }
        ]

    chunks = []
    cursor = 1
    for node in top_defs:
        n_start, n_end = _node_line_span(node)
        if n_start > cursor:
            gap_text = "\n".join(lines[cursor - 1 : n_start - 1])
            if gap_text.strip():
                chunks.append(
                    {
                        "file": norm_path,
                        "symbol": "<module>",
                        "start_line": cursor,
                        "end_line": n_start - 1,
                        "text": gap_text,
                    }
                )

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            chunks.append(
                {
                    "file": norm_path,
                    "symbol": node.name,
                    "start_line": n_start,
                    "end_line": n_end,
                    "text": "\n".join(lines[n_start - 1 : n_end]),
                }
            )
        else:
            sub_defs = [
                child
                for child in node.body
                if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            if not sub_defs:
                chunks.append(
                    {
                        "file": norm_path,
                        "symbol": node.name,
                        "start_line": n_start,
                        "end_line": n_end,
                        "text": "\n".join(lines[n_start - 1 : n_end]),
                    }
                )
            else:
                first_sub_start = _node_line_span(sub_defs[0])[0]
                if first_sub_start > n_start:
                    header_text = "\n".join(lines[n_start - 1 : first_sub_start - 1])
                    if header_text.strip():
                        chunks.append(
                            {
                                "file": norm_path,
                                "symbol": node.name,
                                "start_line": n_start,
                                "end_line": first_sub_start - 1,
                                "text": header_text,
                            }
                        )
                for sub in sub_defs:
                    s_start, s_end = _node_line_span(sub)
                    chunks.append(
                        {
                            "file": norm_path,
                            "symbol": f"{node.name}.{sub.name}",
                            "start_line": s_start,
                            "end_line": s_end,
                            "text": "\n".join(lines[s_start - 1 : s_end]),
                        }
                    )
        cursor = n_end + 1

    if cursor <= len(lines):
        tail_text = "\n".join(lines[cursor - 1 :])
        if tail_text.strip():
            chunks.append(
                {
                    "file": norm_path,
                    "symbol": "<module>",
                    "start_line": cursor,
                    "end_line": len(lines),
                    "text": tail_text,
                }
            )

    return chunks


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

    async def recreate_collection(self, collection_name: str, dim: int = 64) -> bool:
        """Delete the collection if it exists and create a fresh one with dimension `dim`."""
        url = f"{self.base_url}/collections/{collection_name}"
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, transport=self._transport
            ) as client:
                await client.delete(url)
                payload = {
                    "vectors": {
                        "size": dim,
                        "distance": "Cosine",
                    }
                }
                res = await client.put(url, json=payload)
                return res.status_code == 200
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Failed to recreate Qdrant collection %r: %s", collection_name, exc)
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
                if not isinstance(data, dict):
                    return []
                results = data.get("result")
                if isinstance(results, list):
                    return [r for r in results if isinstance(r, dict)]
                return []
        except (httpx.HTTPError, OSError, ValueError) as exc:
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

    async def index_case_files(
        self,
        files: Mapping[str, str],
        embedder: Embedder,
    ) -> str | None:
        """Recreate the per-case collection from the case content hash and index AST chunks."""
        if not await self.is_healthy():
            return None

        col_name = case_collection_name(files)
        seen_paths: set[str] = set()
        chunks: list[dict[str, Any]] = []
        for p, c in sorted(files.items()):
            if not is_modifiable_py_file(p):
                continue
            norm_p = _clean_chunk_path(p)
            if norm_p in seen_paths:
                continue
            seen_paths.add(norm_p)
            chunks.extend(extract_ast_chunks(norm_p, c))

        if not chunks:
            if not await self.recreate_collection(col_name, dim=64):
                return None
            return col_name

        texts = [
            f"{ch['file']}::{ch['symbol']} "
            f"(lines {ch['start_line']}-{ch['end_line']}):\n{ch['text']}"
            for ch in chunks
        ]
        vecs: list[list[float]] = []
        batch_size = 64
        try:
            for idx in range(0, len(texts), batch_size):
                batch_texts = texts[idx : idx + batch_size]
                batch_vecs = await embedder.embed(batch_texts)
                if len(batch_vecs) != len(batch_texts):
                    logger.warning(
                        "Embedder returned %d vectors for %d texts; degrading retrieval",
                        len(batch_vecs),
                        len(batch_texts),
                    )
                    return None
                vecs.extend(batch_vecs)
        except (ProviderError, httpx.HTTPError, OSError, ValueError) as exc:
            logger.info("Embedder failed during case indexing; degrading gracefully: %s", exc)
            return None

        dim = len(vecs[0]) if vecs and vecs[0] else 64
        if not await self.recreate_collection(col_name, dim=dim):
            return None

        points: list[dict[str, Any]] = []
        for i, (ch, v) in enumerate(zip(chunks, vecs, strict=True)):
            points.append(
                {
                    "id": i + 1,
                    "vector": v,
                    "payload": {
                        "file": ch["file"],
                        "symbol": ch["symbol"],
                        "start_line": ch["start_line"],
                        "end_line": ch["end_line"],
                        "text": ch["text"],
                    },
                }
            )
        for idx in range(0, len(points), batch_size):
            if not await self.upsert(col_name, points[idx : idx + batch_size]):
                return None
        return col_name
