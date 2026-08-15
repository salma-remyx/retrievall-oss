from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .core import Chunks, ChunkFilter

__all__ = ["ContextExpand"]


class ContextExpand(ChunkFilter):
    """
    Expand each chunk along the corpus' reading order, merging in the atoms of the
    neighbouring chunks around it.

    Top-k retrieval over fixed-size chunks routinely severs a number from the header
    that gives it meaning — a table row from its unit header, a figure from the fiscal
    year it belongs to. This filter re-derives that context *after* retrieval by
    walking the document's own structure instead of re-chunking it: given the name of
    an existing structural chunking (e.g. `"page"`, `"block"`, `"paragraph"`,
    `"line"`), every chunk is re-cut to span the `before` preceding and `after`
    following chunks in reading order. Being a `ChunkFilter`, it composes with the
    scorers and selectors a corpus already has, so reading "the row *and* its header"
    needs no retrieval machinery beyond what's already there.

    Implementation note (adapted from *Beyond Top-K: Replacing Black-Box Retrieval
    with Interpretable Agentic Operations*, READ, 2026,
    https://arxiv.org/abs/2608.06305): READ's "structural navigation" and "bounded
    span read" operations are folded into a single deterministic filter over the chunk
    graph this framework already maintains, rather than being exposed as MCP tools
    driven by an agent loop. READ's third operation, normalized lexical search, is
    left to the existing `sparsetext` scorers (`Tfidf`, `BM25`) — the paper itself
    reports BM25 as statistically indistinguishable from its own lexical search.

    Parameters
    ----------
    context
        Name of the chunking to navigate along, e.g. `"paragraph"` or `"line"`. Its
        chunks must expose an `ordinal` column.
    before
        How many preceding context chunks to fold in. Defaults to 1.
    after
        How many following context chunks to fold in. Defaults to 1.
    keep_attrs
        Columns of a chunk's *anchor* context chunk (the one its first atom falls in)
        to copy onto the expanded chunk as attributes — e.g. the `page` a hit lives on.

    Examples
    --------
    Re-attach the line a hit's unit header lives on, then read the text back:

    >>> from retrievall.context_expand import ContextExpand
    >>> corpus.chunk("line").filter(ContextExpand("line", before=1, after=0))

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        context: str,
        *,
        before: int = 1,
        after: int = 1,
        keep_attrs: Sequence[str] = (),
    ):
        if before < 0 or after < 0:
            raise ValueError(
                f"`before` and `after` must be non-negative, got {before} and {after}"
            )
        if not (before or after):
            raise ValueError("`before` and `after` cannot both be zero")

        self.context = context
        self.before = before
        self.after = after
        self.keep_attrs = tuple(keep_attrs)

    def _context_order(self, context_chunks: Chunks) -> pl.DataFrame:
        "Context chunks in reading order, with their position along that order."
        columns = pl.from_arrow(context_chunks.chunks).columns
        if "ordinal" not in columns:
            raise ValueError(
                f"Context expansion requires `{self.context}` chunks to have an "
                f"`ordinal` column, but available columns are {columns}"
            )

        order = pl.from_arrow(context_chunks.chunks).select("id", "ordinal")

        if order["ordinal"].null_count():
            raise ValueError(
                f"Context expansion requires `{self.context}` chunks to have "
                "non-null `ordinal` values, but some are null."
            )

        return (
            order.sort("ordinal")
            .with_row_index("pos")
            .with_columns(pl.col("pos").cast(pl.Int64))
        )

    def __call__(self, chunks: Chunks) -> Chunks:
        context_chunks = chunks.corpus.chunk(self.context)
        order = self._context_order(context_chunks)

        available = pl.from_arrow(context_chunks.chunks).columns
        if missing := [attr for attr in self.keep_attrs if attr not in available]:
            raise ValueError(
                f"`keep_attrs` {missing} not found on `{self.context}` chunks; "
                f"available columns are {available}"
            )

        chunk_ids = pl.from_arrow(chunks.chunks).select(pl.col("id").alias("chunk"))

        # No context to navigate along -> nothing to expand into.
        if len(order) == 0:
            return chunks

        # Map every atom to the position of the context chunk covering it.
        atom_pos = (
            pl.from_arrow(context_chunks.chunk_atoms)
            .join(order, left_on="chunk", right_on="id", how="left")
            .select("atom", "pos")
            .drop_nulls()
        )

        # Place each input chunk within the context chunking, via its atoms. A chunk
        # spanning several context chunks anchors to each of them, so it can grow in
        # both directions from every piece of itself.
        anchor = (
            chunk_ids.join(pl.from_arrow(chunks.chunk_atoms), on="chunk", how="left")
            .join(atom_pos, on="atom", how="left")
            .select("chunk", "pos")
            .unique(maintain_order=True)
            .drop_nulls()
        )

        back = [pl.col("pos") - step for step in range(1, self.before + 1)]
        ahead = [pl.col("pos") + step for step in range(1, self.after + 1)]

        # The bounded span: a contiguous window of context positions around each
        # anchor, clipped to the document's edges.
        neighbours = (
            anchor.with_columns(
                lo=pl.min_horizontal(pl.col("pos"), *back),
                hi=pl.max_horizontal(pl.col("pos"), *ahead),
            )
            .with_columns(span=pl.int_ranges(pl.col("lo"), pl.col("hi") + 1))
            .explode("span", empty_as_null=True)
            .with_columns(pl.col("span").clip(0, len(order) - 1))
            .join(order, left_on="span", right_on="pos", how="left")
            .select("chunk", pl.col("id").alias("context"))
            .unique(maintain_order=True)
        )

        expanded = neighbours.join(
            pl.from_arrow(context_chunks.chunk_atoms).rename({"chunk": "ctx_chunk"}),
            left_on="context",
            right_on="ctx_chunk",
            how="inner",
        ).select("chunk", "atom")

        # Chunks whose atoms fall outside the context chunking have no neighbours to
        # expand into; keep their atoms rather than dropping them on the floor.
        unexpanded = chunk_ids.join(
            neighbours.select("chunk").unique(), on="chunk", how="anti"
        ).join(pl.from_arrow(chunks.chunk_atoms), on="chunk", how="inner")

        chunk_atoms = pl.concat([expanded, unexpanded]).unique(maintain_order=True)

        # Carry the anchor's own attributes over, so a hit can be traced back to the
        # part of the document it actually landed in.
        enriched = pl.from_arrow(chunks.chunks)
        if self.keep_attrs:
            carried = (
                anchor.sort("pos")
                .unique(subset="chunk", keep="first", maintain_order=True)
                # `pos` is a row index, so resolve it back to a context chunk ID
                # before picking up that chunk's attributes.
                .join(order.select("pos", "id"), on="pos", how="left")
                .join(
                    pl.from_arrow(context_chunks.chunks).select(
                        "id", *self.keep_attrs
                    ),
                    on="id",
                    how="left",
                )
                .select("chunk", *self.keep_attrs)
            )
            # Drop collisions, mirroring `ChunkDelimitedStringify`'s handling.
            enriched = enriched.drop(
                [attr for attr in self.keep_attrs if attr in enriched.columns]
            ).join(carried, left_on="id", right_on="chunk", how="left")

        return Chunks(
            corpus=chunks.corpus,
            chunks=enriched.to_arrow(),
            chunk_atoms=chunk_atoms.select("chunk", "atom").to_arrow(),
        )
