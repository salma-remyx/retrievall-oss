import numpy as np
import pyarrow as pa
from retrievall.core import AttrExpr, Chunks
from typing import Sequence, Union

__all__ = [
    "ReciprocalRankFusion",
]


ChannelSpec = Union[AttrExpr, "tuple[str, AttrExpr]"]


def _descending_average_ranks(scores: np.ndarray) -> np.ndarray:
    """
    1-based ranks for an array of scores, where the *highest* score gets rank 1.

    Ties share the average of the ranks they span (the same convention as
    ``scipy.stats.rankdata(..., method="average")`` applied to the negated
    scores), so two chunks that a channel cannot distinguish contribute
    identically to the fused score.
    """
    n = scores.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    # Stable descending sort: highest score first, ties keep input order.
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    positions = np.arange(1, n + 1, dtype=np.float64)

    # Tied scores are contiguous after sorting; mark the first row of each tie
    # group and average the 1-based positions within every group.
    is_group_start = np.concatenate(([True], sorted_scores[1:] != sorted_scores[:-1]))
    boundaries = np.flatnonzero(is_group_start)
    sums = np.add.reduceat(positions, boundaries)
    counts = np.diff(np.concatenate((boundaries, [n])))
    avg_by_group = sums / counts

    group_id = np.cumsum(is_group_start) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_by_group[group_id]
    return ranks


class ReciprocalRankFusion(AttrExpr):
    """
    Fuse multiple retrieval channels into a single relevance score per chunk
    using Reciprocal Rank Fusion (RRF).

    Each channel is itself an ``AttrExpr`` scorer (e.g. ``Tfidf`` or ``BM25``)
    that returns one score per chunk, where higher means more relevant. RRF
    converts each channel's scores into ranks and sums the reciprocal ranks,

    .. math:: \\text{rrf}(d) = \\sum_{c} w_c \\cdot \\frac{1}{k + r_c(d)},

    where ``r_c(d)`` is the rank of chunk ``d`` within channel ``c`` (1-based,
    highest score first) and ``w_c`` is that channel's weight. Because the
    fusion operates on *ranks* rather than raw scores, it combines channels
    whose score scales are incomparable (sparse lexical vs. dense semantic vs.
    knowledge-graph) without needing to calibrate them.

    Like ``Tfidf`` and ``BM25``, this is an ``AttrExpr`` that returns one fused
    score per chunk and drops into the same ``.enrich()`` / ``.select()``
    pipeline.

    Implementation note (adapted from APS-RAG, "A corrective agentic hybrid
    RAG and an operations-grounded evaluation for a scientific facility",
    https://arxiv.org/abs/2607.24663): the transferable core of APS-RAG's
    retrieval engine is query-type-adaptive reciprocal-rank fusion of its
    dense, sparse, and knowledge-graph channels. That fusion primitive is
    captured here at full fidelity, with per-channel weighting (the
    ``weights`` parameter) exposing the query-type-adaptive knob. We omit
    APS-RAG's deployed-facility machinery — the corrective agentic loop, the
    MCP/ReAct tooling layer, cross-encoder reranker, and the APS-Bench
    evaluation harness — none of which has a call site in this framework.

    Parameters
    ----------
    channels
        The retrieval channels to fuse. Each entry is either an ``AttrExpr``
        scorer, or a ``(name, AttrExpr)`` tuple where ``name`` is a label used
        only to align entries with ``weights`` (the name is informational).
        At least one channel is required.
    k
        Rank-smoothing constant (>= 0). Larger values dampen the advantage of
        top ranks. Defaults to 60, the value used throughout the IR literature.
    weights
        Optional non-negative per-channel weights, in the same order as
        ``channels``. They are normalized to sum to 1, so a uniform fusion
        (the default) gives every channel equal influence and the fused score
        stays bounded regardless of how many channels are combined. A weight of
        0 drops a channel entirely, letting a caller up-weight the channels it
        trusts for a given query type.
    """

    def __init__(
        self,
        channels: Sequence[ChannelSpec],
        *,
        k: float = 60,
        weights: "Sequence[float] | None" = None,
    ):
        if len(channels) == 0:
            raise ValueError("`channels` must contain at least one scorer.")

        self.channels = [c if isinstance(c, AttrExpr) else c[1] for c in channels]
        self.channel_names = [c if isinstance(c, AttrExpr) else c[0] for c in channels]

        if k < 0:
            raise ValueError(f"`k` must be non-negative, got {k}.")
        self.k = k

        n = len(self.channels)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            weights = list(weights)
            if len(weights) != n:
                raise ValueError(
                    f"`weights` must have one entry per channel "
                    f"({n}), got {len(weights)}."
                )
            if any(w < 0 for w in weights):
                raise ValueError(f"`weights` must be non-negative, got {weights}.")
            if sum(weights) == 0:
                raise ValueError(f"`weights` must not all be zero, got {weights}.")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def __call__(self, chunks: Chunks) -> pa.Array:
        n = len(chunks)
        if n == 0:
            return pa.array([], type=pa.float64())

        fused = np.zeros(n, dtype=np.float64)
        for channel, w in zip(self.channels, self.weights):
            # Each channel is responsible for returning its scores in the
            # correct order for the input chunks.
            scores = np.asarray(channel(chunks).to_pylist(), dtype=np.float64)
            ranks = _descending_average_ranks(scores)
            fused += w / (self.k + ranks)

        return pa.array(fused)
