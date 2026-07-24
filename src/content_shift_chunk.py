"""
LLM-driven content-shift chunking.

Provides :class:`ContentShiftChunk`, a :class:`~retrievall.core.ChunkExpr` that
segments a corpus into variable-size chunks by iteratively asking a large
language model where the content begins to shift.

This is an *adapted port* of **LumberChunker** (Carron et al., 2024,
https://arxiv.org/abs/2406.17526). The paper's core mechanism is kept at full
fidelity: consecutive passages are greedily accumulated into a group up to a
token budget (theta, approximated here by word count), the LLM is asked —
with a short chain-of-thought — to flag the first passage where the content
clearly changes, the chunk ends at that passage, and the scan resumes
sequentially from it. The result is contiguous, non-overlapping, variably
sized segments. Two auxiliary components are substituted to fit Retrievall:

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

    Consecutive atoms (ordered by ``ordinal`` and bounded by ``constrain_to``)
    are greedily accumulated into a group until a word budget is exhausted.
    The LLM is shown the group with incremental ``ID XXXX:`` prefixes and
    asked to flag the first passage where the content clearly changes; the
    atoms up to that passage form a chunk, and the scan resumes from that
    passage. This yields semantically coherent, variably sized,
    non-overlapping segments — the core motivation of LumberChunker.

    Parameters
    ----------
    constrain_to
        Name of the parent chunk that bounds segmentation, e.g. ``"document"``.
    llm
        Callable invoked as ``llm(prompt) -> int | str``. Given the formatted
        prompt (ID-prefixed passages), it should flag the first passage —
        never the first one — where the content clearly changes, answering in
        the paper's ``Answer: ID XXXX`` format, ideally after a brief
        chain-of-thought quoting the shifted passage. A bare integer — the
        0-based index of the first shifted passage — is also accepted, which
        keeps stubs trivial. A response with no usable identifier (e.g.
        ``Answer: ID None``) means "no shift": the whole group stays
        together.
    max_group_words
        Maximum number of words accumulated into one LLM group — the paper's
        token threshold theta, approximated by word count. Default ``660``
        (the paper's best theta of ~550 tokens under its word-count
        approximation). A passage that exceeds the budget on its own forms a
        single-passage chunk.

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        constrain_to: str,
        llm: Callable[[str], object],
        *,
        max_group_words: int = 660,
    ):
        if max_group_words < 1:
            raise ValueError("`max_group_words` must be at least 1.")
        self.constraint = constrain_to
        self.llm = llm
        self.max_group_words = max_group_words

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
        Yield successive atom segments for one constraining chunk.

        Passages are greedily accumulated into a group until adding the next
        would exceed ``max_group_words`` (the paper's token threshold theta,
        approximated by word count). The LLM flags the first passage of the
        group where the content shifts; the chunk ends there and the scan
        resumes sequentially from that passage — never looking back and never
        overlapping.
        """
        i = 0
        n = len(atoms)
        while i < n:
            words = self._word_count(texts[i])
            j = i + 1
            while j < n and words + self._word_count(texts[j]) <= self.max_group_words:
                words += self._word_count(texts[j])
                j += 1
            if j - i == 1:
                # A lone passage that fills the budget on its own cannot be
                # split further; emit it directly without querying the LLM.
                yield atoms[i:j]
                i = j
                continue
            k = self._shift_point(texts[i:j])
            yield atoms[i : i + k]
            i += k

    def _shift_point(self, texts: list) -> int:
        """
        Ask the LLM where the content shifts and return the number of leading
        passages (1..len(texts)) that stay on the same topic.
        """
        raw = self.llm(self._prompt(texts))
        flagged = self._parse_answer(raw)
        if flagged is None:
            return len(texts)  # no shift: the whole group stays together
        # Flagged index 0 (or negative) is invalid per the prompt; clamp to 1
        # so the scan always advances. Indices past the group mean "no shift".
        return max(1, min(len(texts), flagged))

    def _prompt(self, texts: list) -> str:
        passages = "\n".join(f"ID {idx:04d}: {text}" for idx, text in enumerate(texts))
        return (
            "You will be given a group of consecutive passages from a "
            "document. Each passage is prefixed with an incremental "
            "identifier.\n"
            "Find the first passage (not the first one) where the content "
            "clearly changes compared to the passages before it.\n\n"
            f"Passages:\n{passages}\n\n"
            "Briefly explain your reasoning, quoting the passage where the "
            "content shifts and the passage immediately before it. Then, on "
            "a new line, answer with the identifier of the shifted passage "
            "in the format 'Answer: ID XXXX'. If the content does not change "
            "anywhere in the group, answer 'Answer: ID None'."
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _parse_answer(raw: object) -> int | None:
        """
        Extract the 0-based index of the first shifted passage from an LLM
        response, or ``None`` when the response reports no shift.

        The paper's ``Answer: ID XXXX`` contract is preferred; the last match
        wins so chain-of-thought reasoning above the final answer line cannot
        shadow it. Bare integers (from stubs or terse clients) are accepted
        as a fallback.
        """
        if isinstance(raw, bool):  # bool is an int subclass; treat as no answer
            return None
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw) if raw.is_integer() else None
        if not isinstance(raw, str):
            return None
        for pattern in (r"Answer:\s*ID\s*(\d+)", r"ID\s*(\d+)", r"-?\d+"):
            matches = re.findall(pattern, raw, flags=re.IGNORECASE)
            if matches:
                return int(matches[-1])
        return None

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _chunk_id(constraint: object, segment: list) -> int:
        """Deterministic 64-bit id for a (constraint, atom-segment) pair."""
        digest = hashlib.blake2b(digest_size=8)
        digest.update(str(constraint).encode("utf-8"))
        for atom in segment:
            digest.update(b"\x1f")
            digest.update(str(atom).encode("utf-8"))
        return int.from_bytes(digest.digest(), "big")
