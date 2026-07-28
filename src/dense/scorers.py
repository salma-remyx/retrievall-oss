"""Dense (embedding-based) retrieval scorers.

A sibling of ``retrievall.sparsetext``'s ``Tfidf`` / ``BM25``: an ``AttrExpr``
that embeds each chunk and the query and returns one cosine-similarity score
per chunk, dropping into the same ``.enrich()`` / ``.select()`` pipeline.

Adapted from "2D Matryoshka Sentence Embeddings" (Li et al., 2024,
https://arxiv.org/abs/2402.14776v3). The paper's headline retrieval-time
behavior is Matryoshka truncation: a *single* embedding encodes information at
multiple granularities and can be shortened along two axes -- the embedding
``dimensions`` and the encoder ``layers`` -- degrading gracefully to trade
quality for compute without re-embedding. This module ports that mechanism
(truncate -> renormalize -> cosine) over a swappable embedding backend; it does
not reproduce the paper's training.

Implementation mode: adapted port (Mode 2). The 2D-Matryoshka truncation and
scoring mechanism is kept at full fidelity -- both the embedding-dimension axis
and the layer axis -- while the auxiliary components are target-native:

* The learned sentence embedding (a heavy sentence-transformers dependency) is
  isolated behind a lazy import and an injectable ``encoder``, so the core
  package stays slim and the mechanism is usable/testable without the dep.
* The paper's contrastive Matryoshka *training* procedure and its STS/retrieval
  benchmark suite are intentionally out of scope -- there is no training or
  evaluation call site in this framework.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from retrievall.core import AttrExpr, Chunks

__all__ = [
    "Dense",
    "matryoshka_scores",
]


def _normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize ``v`` along ``axis``; zero rows stay zero (no NaN)."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return v / norm


def matryoshka_scores(
    chunk_emb: np.ndarray,
    query_emb: np.ndarray,
    *,
    dimensions: int | None = None,
    layers: int | None = None,
    pool: str = "mean",
) -> np.ndarray:
    """Cosine similarity between ``query_emb`` and each chunk embedding, with
    2D-Matryoshka truncation along the layer and embedding axes.

    Parameters
    ----------
    chunk_emb
        ``(n_chunks, dim)`` or ``(n_chunks, n_layers, dim)``. The optional
        per-text ``n_layers`` axis is the second Matryoshka dimension.
    query_emb
        ``(dim,)`` or ``(n_layers, dim)``.
    dimensions
        First Matryoshka axis: keep the leading ``dimensions`` components of
        every vector, then renormalize. ``None`` keeps the full width.
    layers
        Second Matryoshka axis: when a layer axis is present, keep the *top*
        ``layers`` layers before pooling. ``None`` keeps every layer.
    pool
        How the layer axis collapses to one vector per text: ``"mean"``
        (default) averages the kept layers; ``"last"`` uses only the top layer.

    Returns
    -------
    numpy.ndarray
        ``(n_chunks,)`` float64 cosine similarities in ``[-1, 1]``.

    Notes
    -----
    A single-layer input ``(n, dim)`` is treated as ``(n, 1, dim)`` so both
    Matryoshka axes always apply uniformly.
    """
    chunks = np.asarray(chunk_emb, dtype=np.float64)
    query = np.asarray(query_emb, dtype=np.float64)

    # Promote single-layer inputs to a trivial 3D layer axis.
    if chunks.ndim == 2:
        chunks = chunks[:, np.newaxis, :]
    if query.ndim == 1:
        query = query[np.newaxis, :]

    if chunks.ndim != 3 or query.ndim != 2:
        raise ValueError(
            "`chunk_emb` must be (n, dim) or (n, n_layers, dim) and "
            "`query_emb` must be (dim,) or (n_layers, dim)."
        )

    n_layers = chunks.shape[1]
    if query.shape[0] != n_layers:
        raise ValueError(
            f"query has {query.shape[0]} layers but chunks have {n_layers}."
        )

    # --- 2nd Matryoshka axis: truncate the layer stack to its top layers. ---
    if layers is not None:
        if layers < 1 or layers > n_layers:
            raise ValueError(f"`layers` must be in [1, {n_layers}], got {layers}.")
        chunks = chunks[:, -layers:, :]
        query = query[-layers:, :]

    # Collapse the layer axis into a single vector per text.
    if pool == "mean":
        chunk_vec = chunks.mean(axis=1)
        query_vec = query.mean(axis=0)
    elif pool == "last":
        chunk_vec = chunks[:, -1, :]
        query_vec = query[-1, :]
    else:
        raise ValueError(f"`pool` must be 'mean' or 'last', got {pool!r}.")

    # --- 1st Matryoshka axis: truncate the embedding width, then renormalize. ---
    if dimensions is not None:
        width = chunk_vec.shape[-1]
        if dimensions < 1 or dimensions > width:
            raise ValueError(f"`dimensions` must be in [1, {width}], got {dimensions}.")
        chunk_vec = chunk_vec[..., :dimensions]
        query_vec = query_vec[..., :dimensions]

    # Renormalize *after* truncation: Matryoshka cosine needs unit vectors at
    # the chosen granularity, not the full-width norm.
    chunk_vec = _normalize(chunk_vec, axis=-1)
    query_vec = _normalize(query_vec, axis=-1)

    return chunk_vec @ query_vec


def _default_encoder(model: str, normalize: bool):
    """Build a sentence-transformers encoder returning ``(n, dim)`` vectors.

    Lazily imports ``sentence_transformers`` so importing ``retrievall.dense``
    does not require the heavy dependency. Install it separately, e.g.
    ``pip install sentence-transformers``, or pass ``encoder=`` to ``Dense``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The default `Dense` encoder needs `sentence-transformers`. Install "
            "it with `pip install sentence-transformers`, or pass `encoder=` to "
            "supply your own embedding backend."
        ) from exc

    st = SentenceTransformer(model)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(st.encode(texts, normalize_embeddings=normalize))

    return encode


class Dense(AttrExpr):
    """Add a dense (embedding) cosine-similarity score column to chunks,
    scored against a ``query`` string.

    A direct sibling of ``Tfidf`` / ``BM25``: like them it is an ``AttrExpr``
    that returns one score per chunk and drops into the same ``.enrich()`` /
    ``.select()`` pipeline. The Matryoshka knobs (``dimensions`` and
    ``layers``) are adapted from "2D Matryoshka Sentence Embeddings": a single
    embedding can be shortened at scoring time along its width and (if the
    backend exposes one) its layer stack, degrading gracefully for a
    compute/quality tradeoff without re-embedding.

    Parameters
    ----------
    stringifier
        An ``AttrExpr`` that returns one string per chunk; its output is what
        gets embedded for each chunk.
    query
        Text string chunks are scored against for similarity.
    model
        Sentence-transformers model name for the default encoder. Ignored when
        ``encoder`` is given.
    dimensions
        First Matryoshka axis: truncate every embedding to its leading
        ``dimensions`` components (then renormalize). ``None`` keeps full width.
    layers
        Second Matryoshka axis: keep the top ``layers`` layers when the encoder
        produces a per-layer stack. ``None`` keeps all layers.
    pool
        ``"mean"`` (default) or ``"last"``; how the layer axis is collapsed.
    normalize
        Whether the default sentence-transformers encoder L2-normalizes
        embeddings. Ignored when ``encoder`` is given.
    encoder
        Optional callable ``(list[str]) -> ndarray`` of shape ``(n, dim)`` or
        ``(n, n_layers, dim)``. Plug in a custom backend; also what makes the
        scorer usable (and testable) without the sentence-transformers dep.

    Examples
    --------
    >>> from retrievall.exprs import SimpleStringify
    >>> from retrievall.dense import Dense
    >>> corpus.chunk("page").select(  # doctest: +SKIP
    ...     "ordinal", dense=Dense(SimpleStringify(), query="brown fox")
    ... )
    """

    def __init__(
        self,
        stringifier: AttrExpr,
        query: str,
        *,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int | None = None,
        layers: int | None = None,
        pool: str = "mean",
        normalize: bool = True,
        encoder=None,
    ):
        if dimensions is not None and dimensions < 1:
            raise ValueError(f"`dimensions` must be a positive int, got {dimensions}.")
        if layers is not None and layers < 1:
            raise ValueError(f"`layers` must be a positive int, got {layers}.")

        self.stringifier = stringifier
        self.query = query
        self.model = model
        self.dimensions = dimensions
        self.layers = layers
        self.pool = pool
        self.normalize = normalize
        self.encoder = encoder

    def __call__(self, chunks: Chunks) -> pa.Array:
        # (The stringifier is responsible for returning its strings in the
        # correct order for the input chunks.)
        strings = self.stringifier(chunks).to_pylist()

        encode = (
            self.encoder
            if self.encoder is not None
            else _default_encoder(self.model, self.normalize)
        )

        chunk_emb = np.asarray(encode(strings))
        query_emb = np.asarray(encode([self.query]))[0]

        scores = matryoshka_scores(
            chunk_emb,
            query_emb,
            dimensions=self.dimensions,
            layers=self.layers,
            pool=self.pool,
        )
        return pa.array(scores)
