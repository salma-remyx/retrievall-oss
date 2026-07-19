import numpy as np
import pyarrow as pa
from scipy import sparse
from retrievall.core import AttrExpr, Chunks
from sklearn.feature_extraction.text import CountVectorizer

__all__ = [
    "BM25",
]


class BM25(AttrExpr):
    """
    Add a `bm25` relevance score column to chunks, scored against a `query`
    provided as a text string, using the Okapi BM25 ranking function.

    BM25 is the canonical sibling of TF-IDF for sparse lexical retrieval: like
    `Tfidf`, it is an `AttrExpr` that returns one score per chunk and drops into
    the same `.enrich()` / `.select()` pipeline.

    Implementation note (adapted from BM25S, Xing Han Lù 2024,
    https://arxiv.org/abs/2407.03618): the per-(chunk, term) BM25 contribution
    is computed *eagerly* at scoring time and assembled into a scipy sparse
    matrix, so answering a query reduces to a single sparse matrix-vector
    product. That eager-sparse-scoring trick is BM25S's headline source of
    speedup, expressed here with the numpy/scipy stack the `sparsetext` extra
    already pulls in. To stay within the repo's existing dependencies we reuse
    sklearn's `CountVectorizer` (already required by `sparsetext`) for
    tokenization in place of BM25S's bespoke tokenizer, and we omit BM25S's
    benchmark suite and memory-mapped on-disk index, which have no call site in
    this framework.

    Parameters
    ----------
    stringifier
        An `AttrExpr` that returns one string per chunk; determines how chunks
        are represented as strings for scoring.
    query
        Text string that chunks are scored against.
    k1
        Term-frequency saturation parameter (>= 0). Larger values tolerate more
        repetition. Defaults to 1.5.
    b
        Document-length normalization parameter in [0, 1]. 0 disables length
        normalization; 1 fully normalizes. Defaults to 0.75.
    kwargs
        `CountVectorizer` keyword arguments that control tokenization (e.g.
        `token_pattern`, `ngram_range`, `stop_words`).
    """

    def __init__(
        self,
        stringifier: AttrExpr,
        query: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        **kwargs,
    ):
        if k1 < 0:
            raise ValueError(f"`k1` must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"`b` must be in [0, 1], got {b}")

        self.stringifier = stringifier
        self.query = query
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(**kwargs)

    def __call__(self, chunks: Chunks) -> pa.Array:
        # (The stringifier is responsible for returning its strings
        # in the correct order for the input chunks.)
        strings = self.stringifier(chunks).to_pylist()

        # Term-frequency matrix (n_chunks x n_terms), sparse CSR.
        tf = self.vectorizer.fit_transform(strings).tocsr()
        n_chunks, n_terms = tf.shape

        # No observable terms (empty corpus / all-stopword strings) -> zero
        # scores for every chunk.
        if tf.nnz == 0 or n_terms == 0:
            return pa.array([0.0] * n_chunks, type=pa.float64())

        # Document frequency per term: how many chunks contain each term.
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        # Lucene-style, always-non-negative IDF (BM25S / Lucene default).
        idf = np.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))

        # Per-chunk length (in tokens) and the corpus average.
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean()

        # Eagerly compute the BM25 weight for every nonzero (chunk, term)
        # entry by operating directly on the CSR data array. This is the
        # paper's core: precompute the scoring matrix once, then a query is
        # just a sparse matvec.
        tf_data = tf.data.astype(np.float64)
        # Row index of every nonzero entry, expanded from the CSR indptr.
        row_idx = np.repeat(np.arange(n_chunks), np.diff(tf.indptr))
        col_idx = tf.indices

        if avgdl > 0:
            length_norm = (1.0 - self.b) + self.b * (doc_len[row_idx] / avgdl)
        else:
            length_norm = np.ones_like(tf_data)

        denom = tf_data + self.k1 * length_norm
        weighted = idf[col_idx] * tf_data * (self.k1 + 1.0) / denom

        weights = sparse.csr_matrix((weighted, col_idx, tf.indptr), shape=tf.shape)

        # Encode the query as a term-count vector; scoring is one matvec.
        query_vec = self.vectorizer.transform([self.query])
        scores = (weights @ query_vec.transpose()).toarray().ravel()

        return pa.array(scores)
