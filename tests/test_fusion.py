import pytest

from retrievall.exprs import SimpleStringify
from retrievall.sparsetext import BM25, Tfidf
from retrievall.sparsetext.fusion import ReciprocalRankFusion


class TestReciprocalRankFusion:
    def test_drops_into_select_pipeline(self, ocr_corpus):
        # The fusion AttrExpr plugs into the existing `Chunks.select()` pipeline
        # (retrievall.core) with the same contract as Tfidf/BM25: one score per
        # chunk, materialized as a column.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            fused=ReciprocalRankFusion(
                [
                    Tfidf(SimpleStringify(), query="brown fox"),
                    BM25(SimpleStringify(), query="brown fox"),
                ]
            ),
        )

        # 2 page chunks -> 2 fused scores.
        assert len(res) == 2
        scores = res.column("fused").to_pylist()
        # RRF scores are non-negative, and a uniform 2-channel fusion with k=60
        # tops out at 1/(60+1) when a chunk ranks first in both channels.
        assert all(s >= 0.0 for s in scores)
        assert all(s <= 1.0 / 61.0 + 1e-9 for s in scores)

    def test_drops_into_enrich_pipeline(self, ocr_corpus):
        # Same contract reachable via `.enrich()`.
        corpus = ocr_corpus

        enriched = corpus.chunk("page").enrich(
            fused=ReciprocalRankFusion([BM25(SimpleStringify(), query="brown fox")])
        )

        assert "fused" in enriched.chunks.schema.names
        assert len(enriched.chunks.column("fused")) == 2

    def test_fused_ranking_follows_channels_in_agreement(self, ocr_corpus):
        # "brown" appears only on page 1; both BM25 and Tfidf therefore rank
        # page 1 above page 2 for "brown fox". The fused ranking must agree.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            tfidf=Tfidf(SimpleStringify(), query="brown fox"),
            bm25=BM25(SimpleStringify(), query="brown fox"),
            fused=ReciprocalRankFusion(
                [
                    Tfidf(SimpleStringify(), query="brown fox"),
                    BM25(SimpleStringify(), query="brown fox"),
                ]
            ),
        )

        rows = res.sort_by("ordinal").to_pylist()
        assert rows[0]["fused"] > rows[1]["fused"]
        assert max(rows, key=lambda r: r["fused"])["ordinal"] == 1
        # The fused top matches each individual channel's top.
        assert max(rows, key=lambda r: r["tfidf"])["ordinal"] == 1
        assert max(rows, key=lambda r: r["bm25"])["ordinal"] == 1

    def test_weighting_overrides_a_disagreeing_channel(self, ocr_corpus):
        # Tfidf for "groovy minute" favors page 2 (those words are only on page
        # 2), while BM25 for "brown fox" favors page 1. With all weight on BM25,
        # the fused ranking must follow BM25 and ignore the disagreeing channel.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            bm25=BM25(SimpleStringify(), query="brown fox"),
            fused=ReciprocalRankFusion(
                [
                    Tfidf(SimpleStringify(), query="groovy minute"),
                    BM25(SimpleStringify(), query="brown fox"),
                ],
                weights=[0.0, 1.0],
            ),
        )

        rows = res.to_pylist()
        bm25_order = sorted(rows, key=lambda r: r["bm25"], reverse=True)
        fused_order = sorted(rows, key=lambda r: r["fused"], reverse=True)
        assert [r["ordinal"] for r in fused_order] == [r["ordinal"] for r in bm25_order]

    def test_named_channels(self, ocr_corpus):
        # (name, AttrExpr) tuples label channels and must align with weights.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            fused=ReciprocalRankFusion(
                [
                    ("tfidf", Tfidf(SimpleStringify(), query="brown fox")),
                    ("bm25", BM25(SimpleStringify(), query="brown fox")),
                ],
                weights=[1.0, 1.0],
            ),
        )

        assert len(res) == 2

    def test_params_validate(self):
        with pytest.raises(ValueError):
            ReciprocalRankFusion([])
        with pytest.raises(ValueError):
            ReciprocalRankFusion([BM25(SimpleStringify(), query="x")], k=-1)
        with pytest.raises(ValueError):
            ReciprocalRankFusion(
                [
                    Tfidf(SimpleStringify(), query="x"),
                    BM25(SimpleStringify(), query="x"),
                ],
                weights=[1.0],  # wrong length
            )
        with pytest.raises(ValueError):
            ReciprocalRankFusion([BM25(SimpleStringify(), query="x")], weights=[-1.0])
        with pytest.raises(ValueError):
            ReciprocalRankFusion([BM25(SimpleStringify(), query="x")], weights=[0.0])
