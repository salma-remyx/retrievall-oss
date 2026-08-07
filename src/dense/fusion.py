import pyarrow as pa
from retrievall.core import AttrExpr, Chunks

__all__ = [
    "HybridRRF",
]


class HybridRRF(AttrExpr):
    """
    Fuse two or more chunk scorers into a single hybrid relevance score using
    reciprocal rank fusion (RRF).

    RRF combines ranked lists without needing their raw scores to be on the same
    scale: each scorer contributes `1 / (k + rank)` per chunk (rank 1 is best),
    and the per-chunk contributions are summed. This is the standard way to
    blend a sparse lexical signal (e.g. `BM25` / `Tfidf`) with a dense semantic
    signal (e.g. `DenseEmbedding`) into one ranking.

    Like every other scorer in the framework, `HybridRRF` is an `AttrExpr`
    returning one fused score per chunk, so it drops straight into
    `.enrich()` / `.select()`:

        from retrievall.exprs import SimpleStringify
        from retrievall.sparsetext import BM25
        from retrievall.dense import DenseEmbedding, HybridRRF

        corpus.chunk("page").select(
            "id",
            hybrid=HybridRRF(
                BM25(SimpleStringify(), query="brown fox"),
                DenseEmbedding(SimpleStringify(), query="brown fox"),
            ),
        )

    Adapted from UEmbed (Unified Sparse and Dense Multimodal Embeddings,
    https://arxiv.org/abs/2608.02583v1), whose single model produces both sparse
    and dense representations in one forward pass. Where UEmbed fuses the two
    representations *inside* one model, this combiner fuses any two scorers'
    rankings *at the framework level* — letting the existing sparse scorers and
    a dense scorer behave as one hybrid retriever without a unified model.

    Parameters
    ----------
    *scorers
        One or more `AttrExpr` scorers (e.g. `BM25`, `Tfidf`,
        `DenseEmbedding`). Each is evaluated on the same `Chunks` and its
        per-chunk ranking is fused. At least one is required; two or more is the
        intended hybrid use.
    k
        RRF smoothing constant (>= 1). Larger values dampen the advantage of
        top ranks. Defaults to 60, the value introduced in the original RRF
        paper (Cormack et al., 2009).
    """

    def __init__(self, *scorers: AttrExpr, k: int = 60):
        if not scorers:
            raise ValueError("HybridRRF requires at least one scorer to fuse.")
        if k < 1:
            raise ValueError(f"`k` must be >= 1, got {k}")

        self.scorers = scorers
        self.k = k

    def __call__(self, chunks: Chunks) -> pa.Array:
        n = len(chunks)
        # Accumulate `1 / (k + rank)` contributions from each scorer.
        fused = [0.0] * n

        for scorer in self.scorers:
            scores = scorer(chunks).to_pylist()
            # Rank chunks by this scorer's score, descending; the highest score
            # earns rank 1. `sorted` is stable, so ties keep their input order.
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(order, start=1):
                fused[idx] += 1.0 / (self.k + rank)

        return pa.array(fused, type=pa.float64())
