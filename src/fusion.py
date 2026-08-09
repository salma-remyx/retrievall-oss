from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from .core import AttrExpr, Chunks

__all__ = [
    "WeightedRRF",
]


class WeightedRRF(AttrExpr):
    """
    Fuse multiple per-chunk score columns into a single ranking using
    weighted Reciprocal Rank Fusion (RRF).

    RRF combines several retrievers' ranked lists into one score per chunk
    *without* requiring their raw scores to be on comparable scales: each
    input column is converted to a rank, and the fused score is

        RRF(d) = Σ_i  w_i / (k + rank_i(d))

    where ``rank_i(d)`` is the 1-indexed rank of chunk ``d`` under retriever
    ``i`` (the best chunk for that retriever is rank 1) and ``k`` is a
    smoothing constant (default 60, the value used in the original RRF note
    and common across hybrid retrieval systems). Higher fused scores are
    better, so the result drops straight into the same ``TopK`` /
    ``Threshold`` filters the single-retriever scorers feed.

    This is the literal "hybrid search combining sparse and dense signals"
    primitive: enrich the chunks with one score column per retriever
    (``BM25``/``Tfidf`` today, a dense scorer when one lands), then fuse.

    Adapted from Cormack, Clarke & Büttner, *Reciprocal Rank Fusion (RRF)*
    (2009), as used and empirically stress-tested for weighted multi-retriever
    fusion in Hug et al., *Do Static Embeddings Add Value to Hybrid Dutch
    Retrieval?* (2025, https://arxiv.org/abs/2608.02112v1). That paper's
    portable contribution is the *weighted* RRF combiner and its robust
    two-retriever (lexical + dense) equal-weight default; its evaluation
    harness (simplex weight search via cross-validation, bootstrap CIs,
    sign-randomization tests over MTEB-NL) is downstream evaluation work and
    is intentionally out of scope here.

    Parameters
    ----------
    columns
        Names of the existing score columns on the chunks to fuse, e.g.
        ``["bm25", "tfidf"]``. Each must already have been added with
        ``Chunks.enrich(...)`` before the fusion expression runs.
    weights
        Optional non-negative per-column weights keyed by column name. When
        omitted, every column receives equal weight — the equal-weighting
        default the cited paper found to be robust across its Dutch tasks.
        Provided weights are normalized to sum to one (only their relative
        sizes matter, since a uniform rescale does not change the ranking).
    k
        RRF rank-smoothing constant (>= 0). Larger values dampen the
        advantage of top ranks. Defaults to 60.
    reverse
        Optional per-column flags keyed by column name. ``True`` means lower
        scores are better for that column (e.g. a distance), so the smallest
        value receives rank 1. Defaults to ``False`` for every column —
        higher score is better — matching ``BM25``, ``Tfidf``, and ``TopK``.

    Examples
    --------
    >>> from retrievall.exprs import SimpleStringify
    >>> from retrievall.sparsetext import BM25, Tfidf
    >>> from retrievall.fusion import WeightedRRF
    >>> from retrievall.filters import TopK
    >>> # `corpus` already has rolling page chunks (see README quickstart).
    >>> (
    ...     corpus.chunk("page")
    ...     .enrich(
    ...         bm25=BM25(SimpleStringify(), query="brown fox"),
    ...         tfidf=Tfidf(SimpleStringify(), query="brown fox"),
    ...     )
    ...     .enrich(fused=WeightedRRF(columns=["bm25", "tfidf"]))
    ...     .filter(TopK("fused", 3))
    ...     .select(text=SimpleStringify())
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        columns: list[str],
        *,
        weights: dict[str, float] | None = None,
        k: float = 60.0,
        reverse: dict[str, bool] | None = None,
    ):
        if not columns:
            raise ValueError("`columns` must name at least one score column to fuse.")
        if k < 0:
            raise ValueError(f"`k` must be non-negative, got {k}")

        # Equal weighting when none is supplied; otherwise normalize the
        # supplied weights so their relative sizes — not their absolute
        # scale — drive the fused ranking (a uniform rescale is rank-preserving).
        if weights is None:
            resolved = {c: 1.0 / len(columns) for c in columns}
        else:
            missing = [c for c in columns if c not in weights]
            if missing:
                raise ValueError(
                    f"`weights` is missing an entry for column(s) {missing}."
                )
            negatives = [c for c in columns if weights[c] < 0]
            if negatives:
                raise ValueError(
                    f"`weights` must be non-negative; got negative weight(s) for {negatives}."
                )
            total = sum(weights[c] for c in columns)
            if total <= 0:
                raise ValueError(
                    "`weights` must sum to a positive value; all supplied weights are zero."
                )
            resolved = {c: weights[c] / total for c in columns}

        self.columns = list(columns)
        self.weights = resolved
        self.k = float(k)
        self.reverse = dict(reverse) if reverse else {}

    def __call__(self, chunks: Chunks) -> pa.Array:
        table = chunks.chunks
        available = set(table.schema.names)
        missing = [c for c in self.columns if c not in available]
        if missing:
            raise ValueError(
                f"WeightedRRF could not find score column(s) {missing} on the "
                f"chunks. Available columns: {sorted(available)}. Add them with "
                f"`Chunks.enrich(...)` before fusing."
            )

        n = table.num_rows
        if n == 0:
            return pa.array([], type=pa.float64())

        # Rank each column independently (rank 1 = best chunk) and accumulate
        # the weighted RRF contribution. Polars' `rank(method="min")` assigns
        # tied values the *better* (lower) rank, which is the standard fair
        # treatment for RRF and keeps all-equal columns (e.g. a query term
        # absent everywhere) contribution-neutral.
        scores = pl.from_arrow(table.select(self.columns))
        fused = np.zeros(n, dtype=np.float64)
        for col in self.columns:
            descending = not self.reverse.get(col, False)
            ranks = scores.select(
                pl.col(col).rank(method="min", descending=descending)
            ).to_series()
            fused += self.weights[col] / (self.k + ranks.to_numpy())

        return pa.array(fused)
