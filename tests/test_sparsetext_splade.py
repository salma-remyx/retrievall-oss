import numpy as np
import pytest
from scipy import sparse

from retrievall.exprs import SimpleStringify
from retrievall.sparsetext import Splade


class TestSplade:
    def test_public_export(self):
        # Splade must be reachable from the package root, matching the
        # `BM25`/`Tfidf` convention the README quickstart documents
        # (`from retrievall.sparsetext import ...`).
        import retrievall.sparsetext as sparsetext

        assert "Splade" in sparsetext.__all__
        assert sparsetext.Splade is Splade

    def test_expr(self, ocr_corpus):
        # Drives the existing `Chunks.select()` pipeline (from retrievall.core)
        # with the new scorer, mirroring the Tfidf/BM25 quickstart.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", splade=Splade(SimpleStringify(), query="the")
        )

        # 2 page chunks -> 2 results; "the" appears on both pages.
        assert len(res) == 2
        scores = res.sort_by("ordinal").column("splade").to_pylist()
        assert all(s > 0 for s in scores)

    def test_ranking(self, ocr_corpus):
        # "brown" appears only on page 1; "fox" on both. Page 1 should rank
        # highest for the query "brown fox".
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", splade=Splade(SimpleStringify(), query="brown fox")
        )

        rows = res.to_pylist()
        top = max(rows, key=lambda r: r["splade"])
        bottom = min(rows, key=lambda r: r["splade"])
        assert top["ordinal"] == 1
        assert top["splade"] > bottom["splade"]

    def test_expansion_scores_absent_term(self, ocr_corpus):
        # Core paper contribution: term *expansion*. "sleepy" appears only on
        # page 2, so a statistical-sparse scorer (Tfidf) cannot score page 1 for
        # it; Splade expands page 1's context (fox/dog/the co-occur with
        # "sleepy" on page 2) and assigns page 1 a nonzero score anyway.
        from retrievall.sparsetext import Tfidf

        corpus = ocr_corpus

        splade_res = corpus.chunk("page").select(
            "ordinal", splade=Splade(SimpleStringify(), query="sleepy")
        )
        tfidf_res = corpus.chunk("page").select(
            "ordinal", tfidf=Tfidf(SimpleStringify(), query="sleepy")
        )

        splade = {r["ordinal"]: r["splade"] for r in splade_res.to_pylist()}
        tfidf = {r["ordinal"]: r["tfidf"] for r in tfidf_res.to_pylist()}

        # Tfidf has no signal for an absent term -> exactly 0 on page 1.
        assert tfidf[1] == pytest.approx(0.0)
        # Splade expands -> page 1 gets a positive score despite absence.
        assert splade[1] > 0.0
        # The page that literally contains the term still ranks highest.
        assert splade[2] > splade[1]

    def test_expansion_control_only_prunes(self, ocr_corpus):
        # `max_terms` is the expansion-control budget. Keeping fewer terms per
        # text can only remove (non-negative) weight, so a tighter budget can
        # never raise a chunk's dot-product score.
        corpus = ocr_corpus

        full = corpus.chunk("page").select(
            "ordinal", full=Splade(SimpleStringify(), query="sleepy fox")
        )
        pruned = corpus.chunk("page").select(
            "ordinal",
            pruned=Splade(SimpleStringify(), query="sleepy fox", max_terms=1),
        )

        full_scores = {r["ordinal"]: r["full"] for r in full.to_pylist()}
        pruned_scores = {r["ordinal"]: r["pruned"] for r in pruned.to_pylist()}
        for ordinal in (1, 2):
            assert pruned_scores[ordinal] <= full_scores[ordinal] + 1e-9

    def test_injectable_encoder_is_used(self, ocr_corpus):
        # The learned encoder is an injectable interface (the heavy SPLADE dep
        # stays optional). A custom encoder's sparse vectors must drive scoring
        # directly.
        class ToyEncoder:
            def encode(self, texts):
                n = len(texts)
                # Non-negative vector [1.0, 2.0] per text, independent of
                # content; no topk pruning -> dot([1, 2], [1, 2]) == 5.
                return sparse.csr_matrix(np.tile([1.0, 2.0], (n, 1)))

        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            splade=Splade(SimpleStringify(), query="anything", encoder=ToyEncoder()),
        )

        scores = res.sort_by("ordinal").column("splade").to_pylist()
        assert scores == [pytest.approx(5.0), pytest.approx(5.0)]

    def test_params_validate(self):
        with pytest.raises(ValueError):
            Splade(SimpleStringify(), query="x", max_terms=0)
        with pytest.raises(ValueError):
            Splade(SimpleStringify(), query="x", max_terms=2.5)
        with pytest.raises(ValueError):
            Splade(SimpleStringify(), query="x", expand_strength=-0.5)

    def test_no_match_scores_zero(self, ocr_corpus):
        # A query term absent from the corpus and from any expansion context
        # contributes nothing.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", splade=Splade(SimpleStringify(), query="nonexistentterm")
        )

        scores = res.sort_by("ordinal").column("splade").to_pylist()
        assert scores == [pytest.approx(0.0), pytest.approx(0.0)]
