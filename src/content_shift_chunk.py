"""
LLM-driven content-shift chunking.

Provides :class:`ContentShiftChunk`, a :class:`~retrievall.core.ChunkExpr` that
segments a corpus into variable-size chunks by iteratively asking a large
language model where the content begins to shift.

This is an *adapted port* of **LumberChunker** (Carron et al., 2024,
https://arxiv.org/abs/2406.17526). The paper's core mechanism is kept at full
fidelity: an iterative loop that shows an LLM a group of consecutive passages,
asks for the point where the content shifts, and cuts a chunk up to that point
before continuing. Two auxiliary components are substituted to fit Retrievall:

* The paper segments plain text at *sentence* granularity. This port segments
  at *atom* granularity — the framework's unit of composition — reading each
  atom's ``text`` and ordering by ``ordinal``. Atom granularity is
  user-controlled, so atoms may be tokens, sentences, or paragraphs.
* The LLM is supplied as an injectable ``Callable[[str], object]`` rather than
  a specific model/client, preserving the framework's deliberately slim
  pyarrow/polars/sklearn dependency stack.

The paper's separate evaluation benchmark is intentionally out of scope;
evaluation belongs in a downstream PR.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Iterator

import polars as pl

from .core import ChunkExpr, Chunks, Corpus

__all__ = ["ContentShiftChunk"]


class ContentShiftChunk(ChunkExpr):
    """
    Segment a corpus into variable-size chunks by iteratively asking an LLM
    where the content begins to shift.

    A *group* of ``group_size`` consecutive atoms (ordered by ``ordinal`` and
    bounded by ``constrain_to``) is shown to the LLM. It returns the point at
    which the content shifts; the atoms up to that point form a chunk, and the
    loop continues from the next atom. This yields semantically coherent,
    variably sized segments — the core motivation of LumberChunker.

    Parameters
    ----------
    constrain_to
        Name of the parent chunk that bounds segmentation, e.g. ``"document"``.
    llm
        Callable invoked as ``llm(prompt) -> int | str``. Given the formatted
        prompt (numbered passages), it returns either an integer or a string
        containing one: the number of *leading* passages in the current group
        that stay on the same topic before the content shifts (``1`` to
        ``group_size``). Supplying the LLM as a callable keeps the framework
        dependency-free — wire in any client, a local model, or a stub.
    group_size
        Number of consecutive atoms shown to the LLM per call. Default ``3``.

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        constrain_to: str,
        llm: Callable[[str], object],
        *,
        group_size: int = 3,
    ):
        if group_size < 1:
            raise ValueError("`group_size` must be at least 1.")
        self.constraint = constrain_to
        self.llm = llm
        self.group_size = group_size

    def __call__(self, corpus: Corpus) -> Chunks:
        for col in ("ordinal", "text"):
            if col not in corpus.atoms.schema.names:
                raise KeyError(
                    "Content-shift chunking requires the atoms to have a "
                    f"`{col}` column, but none was found."
                )

        # Atom IDs joined with their text, bounded by the constraining chunk
        # and ordered for a stable reading order. Mirrors the join shape used
        # by the built-in FixedSizeChunk / RegexMatchChunk chunkers.
        ordered = (
            pl.from_arrow(corpus.chunk(self.constraint).chunk_atoms)
            .rename({"chunk": "constraint"})
            .join(pl.from_arrow(corpus.atoms), left_on="atom", right_on="id")
            .with_columns(pl.col("text").fill_null(""))
            .sort(["constraint", "ordinal"])
        )

        # Preserve the corpus's atom-id dtype so downstream joins (e.g.
        # SimpleStringify) line up with `corpus.atoms["id"]`.
        atom_id_dtype = pl.from_arrow(corpus.atoms).schema["id"]

        chunk_rows: list[dict] = []
        chunk_atom_rows: list[dict] = []
        seen_chunks: set[int] = set()

        constraints = ordered.get_column("constraint").unique(maintain_order=True)
        for constraint in constraints.to_list():
            group = ordered.filter(pl.col("constraint") == constraint)
            atoms = group.get_column("atom").to_list()
            texts = group.get_column("text").to_list()

            for segment in self._segments(atoms, texts):
                if not segment:
                    continue
                chunk_id = self._chunk_id(constraint, segment)
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    chunk_rows.append({"id": chunk_id})
                for atom in segment:
                    chunk_atom_rows.append({"chunk": chunk_id, "atom": atom})

        chunks_tbl = pl.DataFrame(chunk_rows, schema={"id": pl.UInt64}).to_arrow()
        chunk_atoms_tbl = pl.DataFrame(
            chunk_atom_rows, schema={"chunk": pl.UInt64, "atom": atom_id_dtype}
        ).to_arrow()

        return Chunks(corpus=corpus, chunks=chunks_tbl, chunk_atoms=chunk_atoms_tbl)

    # -- the iterative content-shift loop ----------------------------------

    def _segments(self, atoms: list, texts: list) -> Iterator[list]:
        """
        Yield successive atom segments for one constraining chunk by repeatedly
        asking the LLM where the content shifts within a group.
        """
        size = self.group_size
        i = 0
        n = len(atoms)
        while i < n:
            group_atoms = atoms[i : i + size]
            group_texts = texts[i : i + size]
            # A trailing group smaller than `group_size` can't be presented as a
            # full window; emit it as the final chunk (LumberChunker does the
            # same with its remaining sentences).
            if len(group_atoms) < size:
                yield group_atoms
                return
            k = self._shift_point(group_texts)
            yield group_atoms[:k]
            i += k

    def _shift_point(self, texts: list) -> int:
        """Ask the LLM for the content-shift point and coerce to a valid index."""
        raw = self.llm(self._prompt(texts))
        return self._coerce_int(raw, default=len(texts), lo=1, hi=len(texts))

    def _prompt(self, texts: list) -> str:
        passages = "\n".join(
            f"[{idx}] {text}" for idx, text in enumerate(texts, start=1)
        )
        return (
            "You are given a group of consecutive passages from a document. "
            "Identify where the content begins to shift to a new topic.\n"
            f"Passages:\n{passages}\n\n"
            "Return a single integer k between 1 and "
            f"{len(texts)}: the number of leading passages that stay on the "
            "same topic before the shift. Return only the integer."
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _coerce_int(raw: object, *, default: int, lo: int, hi: int) -> int:
        """Extract an integer from an LLM response and clamp it to [lo, hi]."""
        if isinstance(raw, bool):  # bool is an int subclass; treat as no answer
            return default
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, float) and raw.is_integer():
            value = int(raw)
        elif isinstance(raw, str):
            match = re.search(r"-?\d+", raw)
            value = int(match.group()) if match else default
        else:
            return default
        return max(lo, min(hi, value))

    @staticmethod
    def _chunk_id(constraint: object, segment: list) -> int:
        """Deterministic 64-bit id for a (constraint, atom-segment) pair."""
        digest = hashlib.blake2b(digest_size=8)
        digest.update(str(constraint).encode("utf-8"))
        for atom in segment:
            digest.update(b"\x1f")
            digest.update(str(atom).encode("utf-8"))
        return int.from_bytes(digest.digest(), "big")
