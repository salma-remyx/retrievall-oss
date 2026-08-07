import pyarrow as pa
from retrievall.core import AttrExpr, Chunks

__all__ = [
    "DenseEmbedding",
]


def _load_default_embedder(model_name: str):
    """
    Lazily build a default embedding backend backed by `sentence-transformers`.

    Kept out of module import time (and therefore out of the slim pyarrow/polars
    core) so that importing `retrievall.dense` stays cheap. Only called when a
    `DenseEmbedding` is constructed without an explicit `embedder`.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "DenseEmbedding needs an embedding backend to score without an "
            "explicit `embedder`. Install one, e.g. "
            "`pip install sentence-transformers`, or pass `embedder=...` with "
            "any callable mapping list[str] -> (n, d) array of vectors."
        ) from exc

    model = SentenceTransformer(model_name)

    def embed(texts):
        # `normalize_embeddings=True` keeps the output on the unit sphere so the
        # dot product in `DenseEmbedding.__call__` is a cosine similarity.
        return model.encode(texts, normalize_embeddings=True)

    return embed


class DenseEmbedding(AttrExpr):
    """
    Add a `dense` relevance score column to chunks by embedding the chunks and
    the `query` and taking their cosine similarity.

    This is the dense counterpart to `Tfidf` / `BM25` in `retrievall.sparsetext`:
    like them, it is an `AttrExpr` returning one score per chunk that drops into
    the same `.enrich()` / `.select()` pipeline.

    Adapted from UEmbed (Unified Sparse and Dense Multimodal Embeddings,
    https://arxiv.org/abs/2608.02583v1). The paper's headline contribution is a
    single decoder-only model that emits both sparse lexical and dense semantic
    representations in one causal forward pass. That unified model is a 2-9B
    parameter multimodal checkpoint the framework cannot host, so this module
    realizes the paper's *separable* contribution here — that adding a dense
    semantic signal alongside the existing sparse scorers enables hybrid
    retrieval — with a pluggable embedder. Pair it with `HybridRRF` (see
    `retrievall.dense.fusion`) to combine sparse and dense rankings the way the
    paper's single model does internally.

    Parameters
    ----------
    stringifier
        An `AttrExpr` that returns one string per chunk; determines how chunks
        are represented as strings for embedding.
    query
        Text string that chunks are scored against for similarity.
    embedder
        Optional callable mapping a `list[str]` to an `(n, d)` array-like of
        vectors (one row per input string). Injecting one keeps the heavy
        embedding backend out of the call site and makes the scorer trivially
        testable offline. If omitted, a `sentence-transformers` backend is
        lazily loaded for `model_name`.
    model_name
        Hugging Face model id used by the default backend. Ignored when
        `embedder` is provided. Defaults to a small, widely available sentence
        embedding model.
    """

    def __init__(
        self,
        stringifier: AttrExpr,
        query: str,
        *,
        embedder=None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.stringifier = stringifier
        self.query = query
        self._embedder = embedder
        self.model_name = model_name

    @property
    def embedder(self):
        # Defer backend construction until the first scoring call so that merely
        # importing / constructing the scorer never triggers a model download.
        if self._embedder is None:
            self._embedder = _load_default_embedder(self.model_name)
        return self._embedder

    def __call__(self, chunks: Chunks) -> pa.Array:
        import numpy as np

        # (The stringifier is responsible for returning its strings
        # in the correct order for the input chunks.)
        strings = self.stringifier(chunks).to_pylist()

        # Embed chunks and query together so they share one vector space; the
        # query is the final row.
        vecs = np.asarray(self.embedder(strings + [self.query]), dtype=np.float64)
        doc_vecs = vecs[:-1]
        query_vec = vecs[-1]

        # L2-normalize for cosine similarity, guarding zero vectors (e.g. an
        # empty chunk) so they score 0 rather than producing NaNs.
        def _unit_normalize(matrix: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return matrix / norms

        doc_unit = _unit_normalize(doc_vecs)
        query_unit = _unit_normalize(query_vec.reshape(1, -1))[0]

        scores = doc_unit @ query_unit
        return pa.array(scores.tolist(), type=pa.float64())
