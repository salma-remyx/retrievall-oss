import hashlib
import re
import sys

import numpy as np
import pyarrow as pa
import pytest

from retrievall import Chunks, Corpus
from retrievall.core import AttrExpr
from retrievall.dense import DenseEmbedding, HybridRRF
from retrievall.dense.scorers import _load_default_embedder
from retrievall.exprs import SimpleStringify
from retrievall.sparsetext import BM25


def _hash_bag_embedder(texts, dim=256):
    """
    Deterministic, dependency-light stand-in for a real embedding model.

    Maps each text to a hashed bag-of-words vector so that texts sharing more
    tokens have higher cosine similarity. Deterministic across runs (uses
    hashlib, not the salted builtin `hash`). Used only to exercise
    `DenseEmbedding` offline — a real backend is injected in production.
    """
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            out[i, bucket] += 1.0
    return out


def _synthetic_chunks(n):
    """A minimal `n`-row `Chunks` for scorer unit tests that ignore contents."""
    corpus = Corpus(pa.table({"id": [], "text": [], "ordinal": []}))
    chunks = pa.table({"id": [f"c{i}" for i in range(n)], "ordinal": list(range(n))})
    chunk_atoms = pa.table({"chunk": [], "atom": []})
    return Chunks(corpus=corpus, chunks=chunks, chunk_atoms=chunk_atoms)


class _FixedScores(AttrExpr):
    """An AttrExpr that ignores its input and returns preset scores."""

    def __init__(self, scores):
        self._scores = scores

    def __call__(self, chunks):  # noqa: D401 - matches AttrExpr contract
        return pa.array(self._scores, type=pa.float64())


class TestDenseEmbedding:
    def test_public_export(self):
        # DenseEmbedding must be reachable from the package root, matching the
        # `Tfidf` / `BM25` convention the README quickstart documents.
        import retrievall.dense as dense

        assert "DenseEmbedding" in dense.__all__
        assert dense.DenseEmbedding is DenseEmbedding

    def test_expr(self, ocr_corpus):
        # Drives the existing `Chunks.select()` pipeline (from retrievall.core)
        # with the dense scorer, mirroring the BM25 quickstart.
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            dense=DenseEmbedding(
                SimpleStringify(), query="brown", embedder=_hash_bag_embedder
            ),
        )

        # 2 page chunks -> 2 results
        assert len(res) == 2

        # "brown" appears only on page 1, so it scores strictly higher there;
        # page 2 shares no token with the query and scores 0.
        rows = res.sort_by("ordinal").to_pylist()
        assert rows[0]["dense"] > rows[1]["dense"]
        assert rows[0]["dense"] > 0
        assert rows[1]["dense"] == pytest.approx(0.0)

    def test_missing_backend_raises_install_hint(self, monkeypatch):
        # Without an explicit embedder and without the backend installed, the
        # scorer must fail with a clear install hint rather than a bare ImportError.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        with pytest.raises(ImportError) as excinfo:
            _load_default_embedder("any-model")

        assert "sentence-transformers" in str(excinfo.value)


class TestHybridRRF:
    def test_public_export(self):
        import retrievall.dense as dense

        assert "HybridRRF" in dense.__all__
        assert dense.HybridRRF is HybridRRF

    def test_requires_scorer(self):
        with pytest.raises(ValueError):
            HybridRRF()

    def test_validates_k(self):
        with pytest.raises(ValueError):
            HybridRRF(_FixedScores([1.0]), k=0)

    def test_rrf_formula(self):
        # Three chunks, two scorers that disagree, so fused ranks are distinct
        # and the RRF sum can be checked by hand (k=60).
        chunks = _synthetic_chunks(3)
        scorer_a = _FixedScores([3.0, 1.0, 2.0])  # ranks: c0=1, c2=2, c1=3
        scorer_b = _FixedScores([2.0, 3.0, 1.0])  # ranks: c1=1, c0=2, c2=3

        fused = HybridRRF(scorer_a, scorer_b, k=60)(chunks).to_pylist()

        expected = [
            1 / 61 + 1 / 62,  # c0: rank 1 from a, rank 2 from b
            1 / 63 + 1 / 61,  # c1: rank 3 from a, rank 1 from b
            1 / 62 + 1 / 63,  # c2: rank 2 from a, rank 3 from b
        ]
        assert fused == pytest.approx(expected)
        # Top chunk is c0.
        assert fused.index(max(fused)) == 0

    def test_hybrid_select_matches_manual_rrf(self, ocr_corpus):
        # The headline integration: fuse the existing sparse `BM25` scorer with
        # the dense scorer through the real `Chunks.select()` pipeline, and
        # confirm the fused score equals an independent manual RRF computation.
        corpus = ocr_corpus
        chunks = corpus.chunk("page")
        bm25 = BM25(SimpleStringify(), query="fox")
        dense = DenseEmbedding(
            SimpleStringify(), query="fox", embedder=_hash_bag_embedder
        )

        # Independent per-chunk scores from each leg.
        bm25_scores = bm25(chunks).to_pylist()
        dense_scores = dense(chunks).to_pylist()

        # Manual reciprocal rank fusion, k=60.
        n = len(chunks)
        expected = [0.0] * n
        for scores in (bm25_scores, dense_scores):
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(order, start=1):
                expected[idx] += 1.0 / (60 + rank)

        res = chunks.select("ordinal", hybrid=HybridRRF(bm25, dense, k=60))

        assert len(res) == 2
        assert res.column("hybrid").to_pylist() == pytest.approx(expected)
        assert all(s > 0 for s in res.column("hybrid").to_pylist())
