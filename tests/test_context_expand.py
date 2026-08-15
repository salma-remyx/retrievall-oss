import pytest
from retrievall.exprs import SimpleStringify
from retrievall.context_expand import ContextExpand
from retrievall.filters import Threshold, TopK


def line_texts(chunks):
    "Materialize each chunk's text, ordered by line `ordinal`."
    return (
        chunks.enrich(text=SimpleStringify())
        .chunks.sort_by("ordinal")["text"]
        .to_pylist()
    )


class TestContextExpand:
    def test_folds_in_neighbouring_lines(self, ocr_corpus):
        # Lines 3 and 4 survive the cut. Line 3 ("Over the") needs the line
        # above it to know what it's talking about.
        res = (
            ocr_corpus.chunk("line")
            .filter(Threshold("ordinal", "<=", 4))
            .filter(ContextExpand("line", before=1, after=0))
        )

        # Chunk count is unchanged (4 input lines)...
        assert len(res) == 4
        # ...but each now carries its preceding line's text too.
        assert line_texts(res) == [
            "The (quick)",
            "The (quick) [brown] fox jumps!",
            "[brown] fox jumps! Over the",
            "Over the <lazy> dog",
        ]

    def test_keeps_chunk_count_and_attrs(self, ocr_corpus):
        # `line` chunks carry a `paragraph` parent but no `page`, so only the
        # former is a legal `keep_attrs` request.
        res = ocr_corpus.chunk("line").filter(
            ContextExpand("line", before=1, after=1, keep_attrs=("paragraph",))
        )

        assert len(res) == len(ocr_corpus.chunk("line"))
        assert "paragraph" in res.chunks.column_names
        assert res.chunks["paragraph"].null_count == 0

    def test_topk_context_kwarg_repairs_the_cut(self, ocr_corpus):
        # The call-site wiring: a top-3 cut over `ordinal` that also keeps the
        # line after each surviving hit. Without `context`, line 1 ("The
        # (quick)") would arrive without the "[brown] fox jumps!" line below it.
        res = ocr_corpus.chunk("line").filter(
            TopK("ordinal", 3, reverse=True, context="line", before=0, after=1)
        )

        assert len(res) == 3
        assert line_texts(res)[0] == "The (quick) [brown] fox jumps!"

    def test_threshold_context_kwarg(self, ocr_corpus):
        res = ocr_corpus.chunk("line").filter(
            Threshold("ordinal", ">", 6, context="line", before=1, after=0)
        )

        # Lines 7 and 8 survive; both gain line 6's text.
        assert line_texts(res) == [
            "minute! dog bounds UPON the",
            "UPON the sleepy fox",
        ]

    def test_clip_at_document_edges(self, ocr_corpus):
        # Line 1 is already at the start, so expanding backwards is a no-op
        # rather than an error.
        res = ocr_corpus.chunk("line").filter(Threshold("ordinal", "<=", 1)).filter(
            ContextExpand("line", before=2, after=0)
        )

        assert len(res) == 1
        assert line_texts(res) == ["The (quick)"]

    def test_default_neighbourhood(self, ocr_corpus):
        # `before`/`after` both default to 1, so the first line is a 2-line window.
        res = ocr_corpus.chunk("line").filter(ContextExpand("line"))

        assert line_texts(res)[0] == "The (quick) [brown] fox jumps!"

    def test_rejects_zero_window(self, ocr_corpus):
        with pytest.raises(ValueError, match="cannot both be zero"):
            ContextExpand("line", before=0, after=0)

    def test_rejects_negative_window(self, ocr_corpus):
        with pytest.raises(ValueError, match="non-negative"):
            ContextExpand("line", before=-1)

    def test_rejects_missing_keep_attrs(self, ocr_corpus):
        with pytest.raises(ValueError, match="not found on `line`"):
            ocr_corpus.chunk("line").filter(
                ContextExpand("line", keep_attrs=("nonexistent",))
            )

    def test_rejects_context_without_ordinal(self, ocr_corpus):
        from retrievall.core import Chunks
        import pyarrow as pa

        corpus = ocr_corpus
        corpus.set_chunk(
            "unordinaled",
            Chunks(
                corpus,
                pa.table({"id": corpus.chunk("line").chunks["id"]}),
                corpus.chunk("line").chunk_atoms,
            ),
        )

        with pytest.raises(ValueError, match="`ordinal` column"):
            corpus.chunk("line").filter(ContextExpand("unordinaled"))
