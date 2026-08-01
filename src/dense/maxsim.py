import re

import numpy as np
import pyarrow as pa
from retrievall.core import AttrExpr, Chunks
from sklearn.feature_extraction.text import HashingVectorizer

__all__ = [
    "MaxSim",
]


# A CountVectorizer/BM25-style word tokenizer so the query and chunk text share
# one notion of "token" with the rest of the `sparsetext` stack.
_TOKEN_RE = re.compile(r"\w\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class MaxSim(AttrExpr):
    """
    Add a `maxsim` late-interaction relevance score column to chunks, scored
    against a `query` provided as a text string using ColBERT-style MaxSim.

    Late-interaction retrieval (Khattab & Zaharia, ColBERT 2020; and the open
    LateOn model of "DenseOn with the LateOn: Fully Open Dense and
    Late-Interaction Models for Multilingual, Long-Context, and Code Search",
    arXiv:2607.27178) represents the query and each chunk as *multi-vector*
    token embeddings and scores a chunk as the sum, over query tokens, of its
    best (max) cosine similarity to any chunk token:

        score(Q, D) = sum_i max_j cos(q_i, d_j)

    That token-level interaction is the mechanism the paper studies: unlike a
    single-vector dense model it keeps per-token resolution, and unlike the
    sparse lexical scorers in `sparsetext` (BM25, Tfidf) the per-token match is
    a graded similarity rather than an exact term hit. `MaxSim` is the dense
    sibling of `BM25` / `Tfidf`: the same `AttrExpr` contract, one score per
    chunk, dropped into the same `.enrich()` / `.select()` pipeline.

    Adapted port (Mode 2). The paper's trained contextual token encoder (the
    LateOn ColBERTv2 backbone) is replaced here with a parameter-free,
    corpus-independent character-n-gram token embedding: each token is mapped
    to an L2-normalized hashed character-n-gram bag via sklearn's stateless
    `HashingVectorizer`, so a token's representation is fixed and does not
    depend on the surrounding corpus (a contextual encoder is also
    corpus-independent). This keeps the MaxSim operator at full fidelity while
    running on the numpy/sklearn stack the `sparsetext` extra already provides
    — no model download and no torch/engine dependency. The paper's training
    recipe, benchmark suite, and translate-train multilingual study are out of
    scope: this is the scoring primitive, not the trained model.

    Parameters
    ----------
    stringifier
        An `AttrExpr` that returns one string per chunk; determines how chunks
        are represented as strings for scoring.
    query
        Text string that chunks are scored against.
    ngram_range
        Character-n-gram range used to embed each token. Defaults to (2, 4),
        which captures subword/morphological overlap so similar (non-identical)
        tokens receive a graded, non-binary match.
    n_features
        Output dimension of the hashing embedding. Larger values reduce hash
        collisions. Defaults to 2 ** 16.
    """

    def __init__(
        self,
        stringifier: AttrExpr,
        query: str,
        *,
        ngram_range: tuple[int, int] = (2, 4),
        n_features: int = 2**16,
    ):
        if n_features <= 0:
            raise ValueError(f"`n_features` must be positive, got {n_features}")
        if not (1 <= ngram_range[0] <= ngram_range[1]):
            raise ValueError(
                f"`ngram_range` must satisfy 1 <= low <= high, got {ngram_range}"
            )

        self.stringifier = stringifier
        self.query = query
        self.ngram_range = ngram_range
        self.n_features = n_features
        # Stateless token embedder: one L2-normalized char-n-gram vector per
        # token. `norm="l2"` makes every dot product a cosine similarity, and
        # `alternate_sign=False` keeps the graded similarity a clean (non-negative)
        # n-gram overlap so identical tokens score 1.0 and disjoint tokens 0.0.
        self._embedder = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            norm="l2",
            alternate_sign=False,
        )

    def __call__(self, chunks: Chunks) -> pa.Array:
        # (The stringifier is responsible for returning its strings in the
        # correct order for the input chunks.)
        strings = self.stringifier(chunks).to_pylist()
        query_tokens = _tokenize(self.query)
        n_chunks = len(strings)

        # No query tokens -> nothing to match -> zero score for every chunk.
        if not query_tokens:
            return pa.array([0.0] * n_chunks, type=pa.float64())

        # Embed query tokens once: (n_q, n_features), rows already unit-norm.
        q_vecs = self._embedder.transform(query_tokens)

        scores = np.zeros(n_chunks, dtype=np.float64)
        for i, text in enumerate(strings):
            tokens = _tokenize(text)
            # An empty chunk has no tokens to match against -> score stays 0.
            if not tokens:
                continue

            d_vecs = self._embedder.transform(tokens)  # (n_d, n_features)
            # Cosine similarity between every query token and every chunk token;
            # both sides are L2-normalized, so the matmul is cosines directly.
            sims = (q_vecs @ d_vecs.transpose()).toarray()  # (n_q, n_d)
            # MaxSim: the best chunk token per query token, summed over the query.
            scores[i] = float(sims.max(axis=1).sum())

        return pa.array(scores)
