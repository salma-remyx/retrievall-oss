import math

import pytest
from retrievall import WeightedRRF  # wired into the package root (src/__init__.py)
from retrievall.exprs import SimpleStringify
from retrievall.filters import TopK
from retrievall.sparsetext import BM25, Tfidf


def _ranks(scores):
    """1-indexed 'min' ranks: the best (highest) value gets rank 1, ties share it."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranks = [0] * len(scores)
    for position, i in enumerate(order):
        ranks[i] = position + 1
    return ranks


class TestWeightedRRF:
    def test_public_export(self):
        # WeightedRRF is reachable from the package root, mirroring how
        # `Chunks`/`Corpus` are re-exported from `retrievall.core`.
        import retrievall

        assert "WeightedRRF" in retrievall.__all__
        assert retrievall.WeightedRRF is WeightedRRF

    def test_fused_ranking_prefers_better_chunk(self, ocr_corpus):
        # "brown" appears only on page 1; "fox" on both. Both BM25 and TF-IDF
        # rank page 1 higher for "brown fox", so the fused score must too.
        corpus = ocr_corpus

        res = (
            corpus.chunk("page")
            .enrich(
                bm25=BM25(SimpleStringify(), query="brown fox"),
                tfidf=Tfidf(SimpleStringify(), query="brown fox"),
            )
            .enrich(fused=WeightedRRF(columns=["bm25", "tfidf"]))
            .select("ordinal", "bm25", "tfidf", "fused")
        )

        rows = res.sort_by("ordinal").to_pylist()
        assert len(rows) == 2
        assert all(math.isfinite(r["fused"]) for r in rows)

        top = max(rows, key=lambda r: r["fused"])
        assert top["ordinal"] == 1

    def test_rrf_value_matches_formula(self, ocr_corpus):
        # The fused score must equal the weighted-RRF of the input columns'
        # ranks: 0.5/(k+rank_bm25) + 0.5/(k+rank_tfidf), with k=60.
        corpus = ocr_corpus
        k = 60.0

        res = (
            corpus.chunk("page")
            .enrich(
                bm25=BM25(SimpleStringify(), query="brown fox"),
                tfidf=Tfidf(SimpleStringify(), query="brown fox"),
            )
            .enrich(fused=WeightedRRF(columns=["bm25", "tfidf"], k=k))
            .select("bm25", "tfidf", "fused")
            .to_pylist()
        )

        bm25_ranks = _ranks([r["bm25"] for r in res])
        tfidf_ranks = _ranks([r["tfidf"] for r in res])
        expected = [
            0.5 / (k + bm25_ranks[i]) + 0.5 / (k + tfidf_ranks[i])
            for i in range(len(res))
        ]

        assert [r["fused"] for r in res] == pytest.approx(expected, rel=1e-9)

    def test_pipeline_filters_on_fused_score(self, ocr_corpus):
        # The fused column feeds the existing TopK filter end-to-end, exactly
        # like the single-scorer quickstart in the README.
        corpus = ocr_corpus

        res = (
            corpus.chunk("page")
            .enrich(
                bm25=BM25(SimpleStringify(), query="brown fox"),
                tfidf=Tfidf(SimpleStringify(), query="brown fox"),
            )
            .enrich(fused=WeightedRRF(columns=["bm25", "tfidf"]))
            .filter(TopK("fused", 1))
            .select("ordinal", text=SimpleStringify())
        )

        rows = res.to_pylist()
        assert len(rows) == 1
        assert rows[0]["ordinal"] == 1

    def test_equal_weight_default_matches_explicit(self, ocr_corpus):
        corpus = ocr_corpus
        enrich = corpus.chunk("page").enrich(
            bm25=BM25(SimpleStringify(), query="the"),
            tfidf=Tfidf(SimpleStringify(), query="the"),
        )

        implicit = enrich.enrich(fused=WeightedRRF(columns=["bm25", "tfidf"]))
        explicit = enrich.enrich(
            fused=WeightedRRF(
                columns=["bm25", "tfidf"], weights={"bm25": 1, "tfidf": 1}
            )
        )

        assert implicit.select("fused").column("fused").to_pylist() == pytest.approx(
            explicit.select("fused").column("fused").to_pylist()
        )

    def test_missing_column_raises(self, ocr_corpus):
        corpus = ocr_corpus

        with pytest.raises(ValueError, match="could not find score column"):
            (
                corpus.chunk("page")
                .enrich(bm25=BM25(SimpleStringify(), query="fox"))
                .enrich(fused=WeightedRRF(columns=["bm25", "dense"]))
                .select("fused")
            )

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="at least one"):
            WeightedRRF(columns=[])
        with pytest.raises(ValueError, match="non-negative"):
            WeightedRRF(columns=["a", "b"], k=-1)
        with pytest.raises(ValueError, match="missing an entry"):
            WeightedRRF(columns=["a", "b"], weights={"a": 1.0})
        with pytest.raises(ValueError, match="non-negative"):
            WeightedRRF(columns=["a", "b"], weights={"a": 1.0, "b": -0.5})
