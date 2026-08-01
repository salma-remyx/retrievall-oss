import pytest
from retrievall.dense import MaxSim
from retrievall.exprs import SimpleStringify


class TestMaxSim:
    def test_public_export(self):
        # MaxSim must be reachable from the package root, matching the `BM25`
        # / `Tfidf` convention the README quickstart documents
        # (`from retrievall.<module> import ...`).
        import retrievall.dense as dense

        assert "MaxSim" in dense.__all__
        assert dense.MaxSim is MaxSim

    def test_expr(self, ocr_corpus):
        # Drives the existing `Chunks.select()` pipeline (from retrievall.core)
        # with the new scorer, mirroring the BM25 quickstart.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", maxsim=MaxSim(SimpleStringify(), query="the")
        )

        # 2 page chunks -> 2 results
        assert len(res) == 2

        # A single query token ("the") that both pages contain scores 1.0 each:
        # MaxSim over one query token is the best cosine to any chunk token,
        # and an identical token is a perfect (cosine = 1.0) match.
        rows = res.sort_by("ordinal").to_pylist()
        assert [r["maxsim"] for r in rows] == [
            pytest.approx(1.0, abs=1e-6),
            pytest.approx(1.0, abs=1e-6),
        ]
        assert all(r["maxsim"] > 0 for r in rows)

    def test_ranking(self, ocr_corpus):
        # "brown" appears only on page 1; "fox" on both. Page 1 should rank
        # highest for the query "brown fox" — same ranking signal as BM25, but
        # produced by token-level late interaction rather than term statistics.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", maxsim=MaxSim(SimpleStringify(), query="brown fox")
        )

        rows = res.to_pylist()
        top = max(rows, key=lambda r: r["maxsim"])
        bottom = min(rows, key=lambda r: r["maxsim"])

        assert top["ordinal"] == 1
        assert top["maxsim"] > bottom["maxsim"]

    def test_fuzzy_match_graded(self, ocr_corpus):
        # The dense, graded bit: "jumping" is not an exact term in the corpus,
        # but page 1 holds "jumps", which shares character n-grams with it.
        # MaxSim therefore returns a partial (0 < score < 1) match on page 1,
        # ranking it above page 2 (no morphologically related token). A pure
        # exact-match lexical scorer would score this as zero.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", maxsim=MaxSim(SimpleStringify(), query="jumping")
        )

        rows = res.sort_by("ordinal").to_pylist()
        page1, page2 = rows[0]["maxsim"], rows[1]["maxsim"]

        assert 0.0 < page1 < 1.0
        assert page1 > page2

    def test_empty_query_scores_zero(self, ocr_corpus):
        # A query with no tokens has nothing to match -> zero for every chunk.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            maxsim=MaxSim(SimpleStringify(), query="! @# $%")
        )

        scores = res.column("maxsim").to_pylist()
        assert scores == [pytest.approx(0.0), pytest.approx(0.0)]
