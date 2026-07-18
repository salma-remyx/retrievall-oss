import pytest
from retrievall.exprs import SimpleStringify
from retrievall.sparsetext.bm25 import BM25


class TestBM25:
    def test_expr(self, ocr_corpus):
        # Drives the existing `Chunks.select()` pipeline (from retrievall.core)
        # with the new scorer, mirroring the Tfidf quickstart.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", bm25=BM25(SimpleStringify(), query="the")
        )

        # 2 page chunks -> 2 results
        assert len(res) == 2

        # Both pages contain "the" twice and have equal token length, so their
        # BM25 scores are equal and positive.
        rows = res.sort_by("ordinal").to_pylist()
        assert [r["bm25"] for r in rows] == [
            pytest.approx(0.2605, abs=1e-3),
            pytest.approx(0.2605, abs=1e-3),
        ]
        assert all(r["bm25"] > 0 for r in rows)

    def test_ranking(self, ocr_corpus):
        # "brown" appears only on page 1; "fox" on both. Page 1 should rank
        # highest for the query "brown fox".
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", bm25=BM25(SimpleStringify(), query="brown fox")
        )

        rows = res.to_pylist()
        top = max(rows, key=lambda r: r["bm25"])
        bottom = min(rows, key=lambda r: r["bm25"])

        assert top["ordinal"] == 1
        assert top["bm25"] > bottom["bm25"]

    def test_params_validate(self):
        with pytest.raises(ValueError):
            BM25(SimpleStringify(), query="x", k1=-1)
        with pytest.raises(ValueError):
            BM25(SimpleStringify(), query="x", b=1.5)

    def test_no_match_scores_zero(self, ocr_corpus):
        # A query term absent from the corpus contributes nothing.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            bm25=BM25(SimpleStringify(), query="nonexistentterm")
        )

        scores = res.column("bm25").to_pylist()
        assert scores == [pytest.approx(0.0), pytest.approx(0.0)]
