from .context_expand import ContextExpand
from .core import Chunks, ChunkFilter
import pyarrow as pa
import pyarrow.compute  # Registers the `pa.compute` namespace used below

from typing import Collection, Literal

__all__ = ["TopK", "Threshold", "EqualTo"]

# Both selectors below take a hard truncation of the ranked list, which is exactly
# where a top-k chunk boundary severs a number from the header that gives it its
# meaning. Each exposes a `context` kwarg that repairs the cut afterwards by folding
# the neighbouring `context` chunks back in — the structural-navigation half of READ
# (Beyond Top-K, 2026, arXiv:2608.06305), expressed as a filter a pipeline already
# knows how to apply.


class TopK(ChunkFilter):
    """
    Selects the top `k` chunks from a given collection of chunks, based on
    a given attribute's values.

    Parameters
    ----------
    attr
        Attribute to filter by
    k
        Number of items to select
    reverse
        If `True`, return *bottom* `k` items instead.
    context
        Name of a chunking to fold in around each surviving chunk, so a hit keeps
        the lines around it rather than the `k` the cut happened to allow. Off by
        default. See `ContextExpand`.

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        attr: str,
        k: int,
        reverse: bool = False,
        *,
        context: str | None = None,
        before: int = 1,
        after: int = 1,
    ):
        self.attr = attr
        self.k = k
        self.reverse = reverse
        self.context = context
        self.before = before
        self.after = after

    def __call__(self, chunks: Chunks) -> Chunks:
        values = chunks.chunks[self.attr]

        filtered_idxs = pa.compute.select_k_unstable(
            values,
            k=self.k,
            sort_keys=[(self.attr, "ascending" if self.reverse else "descending")],
        )

        # Convert indices to a boolean mask
        filtered_chunks = chunks.chunks.take(filtered_idxs)

        res = Chunks(
            corpus=chunks.corpus, chunks=filtered_chunks, chunk_atoms=chunks.chunk_atoms
        )

        if self.context is not None:
            res = res.filter(
                ContextExpand(self.context, before=self.before, after=self.after)
            )

        return res


class Threshold(ChunkFilter):
    """
    Selects all chunks with an attribute above a certain threshold

    Parameters
    ----------
    attr
        Attribute to filter by
    direction : {">", "<"}
        ">" to select chunks that are greater than the threshold, "<" to
        select chunks that are less than the threshold.
    threshold
        Value of threshold
    context
        Name of a chunking to fold in around each surviving chunk, so a hit keeps
        the lines around it rather than wherever the cut happened to land. Off by
        default. See `ContextExpand`.

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        attr: str,
        direction: Literal[">", ">=", "<", "<="],
        value,
        *,
        context: str | None = None,
        before: int = 1,
        after: int = 1,
    ):
        if direction not in {">", ">=", "<", "<="}:
            raise ValueError('`direction` must one of: ">", ">=", "<", "<="')

        self.attr = attr
        self.threshold = value
        self.direction = direction
        self.context = context
        self.before = before
        self.after = after

    def __call__(self, chunks: Chunks) -> Chunks:
        values = chunks.chunks[self.attr]

        match self.direction:
            case ">=":
                chunk_mask = pa.compute.greater_equal(values, self.threshold)
            case ">":
                chunk_mask = pa.compute.greater(values, self.threshold)
            case "<=":
                chunk_mask = pa.compute.less_equal(values, self.threshold)
            case "<":
                chunk_mask = pa.compute.less(values, self.threshold)
            case _:
                raise ValueError(
                    f"'`direction` must one of: '>', '>=', '<', '<='. Got `{self.direction}`"
                )

        res = Chunks(
            corpus=chunks.corpus,
            chunks=chunks.chunks.filter(chunk_mask),
            chunk_atoms=chunks.chunk_atoms,
        )

        if self.context is not None:
            res = res.filter(
                ContextExpand(self.context, before=self.before, after=self.after)
            )

        return res


class EqualTo(ChunkFilter):
    """
    Selects all chunks with an attribute that is equal to *any* value
    in a collection of values.

    Note that `values` is a collection; if you want "equal to a single value",
    provide a single-item collection to the `values` argument.

    Parameters
    ----------
    attr
        Attribute to filter by
    values
        Values to filter by. *Must* be a collection.

    Returns
    -------
    Chunks
    """

    def __init__(self, attr: str, values: Collection):
        self.attr = attr
        self.values = values

    def __call__(self, chunks: Chunks) -> Chunks:
        values = chunks.chunks[self.attr]

        chunk_mask = pa.compute.is_in(values, pa.array(self.values))

        return Chunks(
            corpus=chunks.corpus,
            chunks=chunks.chunks.filter(chunk_mask),
            chunk_atoms=chunks.chunk_atoms,
        )
