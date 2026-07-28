import re
import sys
import zlib

import numpy as np
import pytest

from retrievall.core import AttrExpr
from retrievall.dense import Dense, matryoshka_scores
from retrievall.exprs import SimpleStringify


def _word_encoder(dim: int = 1024, n_layers: int = 3):
    """Deterministic, dependency-free bag-of-words encoder.

    Returns an ``(n, n_layers, dim)`` array so the Matryoshka *layer* axis is
    exercised too. Each "layer" is a fixed scaling of the same normalized
    word-bag vector, standing in for a multi-layer encoder without requiring
    sentence-transformers or any network access. Words are hashed with
    ``zlib.crc32`` (stable across processes / PYTHONHASHSEED) and a large
    ``dim`` keeps collisions negligible for the tiny test vocabularies.
    """

    def encode(texts):
        out = np.zeros((len(texts), n_layers, dim), dtype=np.float64)
        for i, text in enumerate(texts):
            base = np.zeros(dim, dtype=np.float64)
            for token in re.findall(r"\w+", text.lower()):
                base[zlib.crc32(token.encode()) % dim] += 1.0
            norm = np.linalg.norm(base)
            if norm > 0:
                base = base / norm
            for layer in range(n_layers):
                out[i, layer] = base * (1.0 + 0.1 * layer)
        return out

    return encode


class TestPublicSurface:
    def test_is_attr_expr(self):
        # Dense must satisfy the verified AttrExpr scorer contract (from the
        # existing retrievall.core module), just like Tfidf / BM25.
        assert issubclass(Dense, AttrExpr)

    def test_public_export(self):
        import retrievall.dense as dense

        assert "Dense" in dense.__all__
        assert dense.Dense is Dense


class TestPipelineIntegration:
    def test_dense_via_select(self, ocr_corpus):
        # Drives the existing Chunks.select() pipeline (retrievall.core) with
        # the new scorer, mirroring the Tfidf/BM25 quickstart.
        corpus = ocr_corpus
        res = corpus.chunk("page").select(
            "ordinal",
            dense=Dense(SimpleStringify(), query="brown fox", encoder=_word_encoder()),
        )

        # 2 page chunks -> 2 results, each a finite cosine score in [-1, 1].
        assert len(res) == 2
        rows = res.sort_by("ordinal").to_pylist()
        scores = [r["dense"] for r in rows]
        assert all(np.isfinite(scores))
        assert all(-1.0 <= s <= 1.0 for s in scores)

    def test_ranking(self, ocr_corpus):
        # "brown" appears only on page 1; "fox" on both. Page 1 shares both
        # query words with "brown fox", page 2 only one -> page 1 ranks highest.
        corpus = ocr_corpus
        res = corpus.chunk("page").select(
            "ordinal",
            dense=Dense(SimpleStringify(), query="brown fox", encoder=_word_encoder()),
        )
        rows = res.to_pylist()
        top = max(rows, key=lambda r: r["dense"])
        bottom = min(rows, key=lambda r: r["dense"])
        assert top["ordinal"] == 1
        assert top["dense"] > bottom["dense"]

    def test_dimensions_knob_runs(self, ocr_corpus):
        # The Matryoshka `dimensions` knob must run end-to-end through the
        # pipeline and stay bounded; we assert no ranking claim here because a
        # hash bag-of-words is not Matryoshka-trained (graceful degradation
        # under truncation is pinned by a structured-embedding test below).
        corpus = ocr_corpus
        res = corpus.chunk("page").select(
            "ordinal",
            dense=Dense(
                SimpleStringify(),
                query="brown fox",
                dimensions=8,
                encoder=_word_encoder(),
            ),
        )
        rows = res.to_pylist()
        assert len(rows) == 2
        assert all(-1.0 <= r["dense"] <= 1.0 for r in rows)
        assert all(np.isfinite(r["dense"]) for r in rows)


class TestMatryoshkaDimensionAxis:
    def test_truncation_changes_score(self):
        # Truncating the embedding width changes the cosine score -- the
        # Matryoshka knob does something, not a no-op.
        chunk = np.array([[1.0, 1.0, 0.0, 0.0]])  # (1, 4)
        query = np.array([1.0, 0.0, 0.0, 0.0])

        # Full width: chunk -> [0.7071, 0.7071, 0, 0]; cosine vs [1,0,0,0].
        full = matryoshka_scores(chunk, query)
        assert full[0] == pytest.approx(1 / np.sqrt(2), abs=1e-9)

        # Truncate to 1 component: chunk -> [1]; query -> [1]; cosine = 1.0.
        trunc = matryoshka_scores(chunk, query, dimensions=1)
        assert trunc[0] == pytest.approx(1.0, abs=1e-9)
        assert trunc[0] != pytest.approx(full[0], abs=1e-6)

    def test_graceful_degradation_preserves_ranking(self):
        # The paper's headline claim: when the embedding carries the ranking
        # signal in its LEADING components (Matryoshka structure), truncating
        # toward the front degrades gracefully and preserves the ranking.
        # chunk_a matches the query in the leading dims; chunk_b differs there;
        # the trailing dims are shared noise.
        query = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
        chunk_a = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
        chunk_b = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        chunks = np.stack([chunk_a, chunk_b])

        for dims in (None, 4, 3, 2):
            scores = matryoshka_scores(chunks, query, dimensions=dims)
            assert scores[0] > scores[1], f"ranking broke at dimensions={dims}"


class TestMatryoshkaLayerAxis:
    def test_pool_and_layer_truncation(self):
        # chunk layers [[1,0],[0,1]]; query layers [[0,1],[1,0]] (swapped).
        chunk = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
        query = np.array([[0.0, 1.0], [1.0, 0.0]])  # (2, 2)

        # Mean-pool over layers: both collapse to [0.5, 0.5] -> identical -> 1.0.
        assert matryoshka_scores(chunk, query, pool="mean")[0] == pytest.approx(
            1.0, abs=1e-9
        )
        # Top (last) layer: chunk top=[0,1], query top=[1,0] -> orthogonal -> 0.
        assert matryoshka_scores(chunk, query, pool="last")[0] == pytest.approx(
            0.0, abs=1e-9
        )
        # Truncating to the top layer then mean-pooling == top layer -> 0.0.
        assert matryoshka_scores(chunk, query, layers=1, pool="mean")[
            0
        ] == pytest.approx(0.0, abs=1e-9)


class TestEdgeCasesAndValidation:
    def test_zero_vector_safe(self):
        # A zero embedding must not produce NaN; cosine is 0.
        chunk = np.array([[0.0, 0.0], [1.0, 0.0]])
        query = np.array([0.0, 0.0])
        scores = matryoshka_scores(chunk, query)
        assert np.all(np.isfinite(scores))
        assert scores.tolist() == [0.0, 0.0]

    def test_matryoshka_validation(self):
        chunk = np.zeros((2, 4))
        query = np.zeros(4)
        with pytest.raises(ValueError):
            matryoshka_scores(chunk, query, dimensions=0)
        with pytest.raises(ValueError):
            matryoshka_scores(chunk, query, dimensions=5)  # > width
        with pytest.raises(ValueError):
            matryoshka_scores(chunk, query, pool="median")
        with pytest.raises(ValueError):
            matryoshka_scores(np.zeros((1, 3, 4)), np.zeros((2, 4)))  # layer mismatch

    def test_dense_param_validation(self):
        with pytest.raises(ValueError):
            Dense(SimpleStringify(), query="x", dimensions=0)
        with pytest.raises(ValueError):
            Dense(SimpleStringify(), query="x", dimensions=-3)
        with pytest.raises(ValueError):
            Dense(SimpleStringify(), query="x", layers=0)

    def test_default_encoder_missing_dep(self, ocr_corpus, monkeypatch):
        # Without an explicit `encoder`, Dense falls back to sentence-transformers.
        # When that dep is absent it must raise a helpful ImportError (not crash
        # opaquely). Force the absence deterministically regardless of environment.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        scorer = Dense(SimpleStringify(), query="brown fox")
        with pytest.raises(ImportError, match="sentence-transformers"):
            ocr_corpus.chunk("page").select(dense=scorer)
