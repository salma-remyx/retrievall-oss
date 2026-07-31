import numpy as np
import pyarrow as pa
from scipy import sparse
from retrievall.core import AttrExpr, Chunks
from sklearn.feature_extraction.text import CountVectorizer

__all__ = [
    "Splade",
    "SpladeEncoder",
]


def _activate(weights):
    """SPLADE-style activation of a sparse matrix: ``log(1 + ReLU(x))``.

    Non-negative and magnitude-dampened; negative entries are floored to zero
    and pruned so the result stays genuinely sparse.
    """
    out = weights.copy().tocsr()
    out.data = np.log1p(np.maximum(out.data, 0.0))
    out.eliminate_zeros()
    return out


def _topk_per_row(matrix, k):
    """Expansion control: keep only the ``k`` largest-weight terms per row.

    This is the inference-time realization of the paper's probabilistic
    expansion budget (an expected cap on the number of active terms).
    ``k=None`` leaves the matrix unchanged.
    """
    if k is None:
        return matrix
    lil = matrix.tolil(copy=True)
    for i in range(lil.shape[0]):
        data, cols = lil.data[i], lil.rows[i]
        if len(data) > k:
            keep = np.argpartition(data, -k)[-k:]
            lil.data[i] = [data[j] for j in keep]
            lil.rows[i] = [cols[j] for j in keep]
    return lil.tocsr()


class Splade(AttrExpr):
    """
    Add a ``splade`` learned-sparse relevance score column to chunks, scored
    against a ``query`` using sparse lexical vectors with controlled term
    expansion and sparse dot-product scoring.

    Splade ("Sparse Lexical and Expansion") is the learned-sparse sibling of
    ``Tfidf``/``BM25``: like them it is an ``AttrExpr`` returning one score per
    chunk that drops into the same ``.enrich()`` / ``.select()`` pipeline.
    Unlike the statistical-sparse scorers, its per-text vector is not limited
    to the terms literally present -- it *expands* onto semantically related
    vocabulary terms, keeps only a controlled budget of them, and scores by
    sparse dot product (the same inverted-index-friendly path as BM25/TF-IDF).

    Implementation note (Mode 2 adapted port of "Multimodal Learned Sparse
    Retrieval with Probabilistic Expansion Control", Bhojanapalli et al. 2024,
    https://arxiv.org/abs/2402.17535): the paper's *inference-time* mechanism
    is kept at full fidelity -- a per-text sparse lexical vector built with
    SPLADE-style ``log(1 + ReLU(logits))`` activation, pooled over token
    positions, pruned by a top-k *expansion-control* budget (the inference-time
    form of the paper's expected-active-term control), and scored by sparse dot
    product. Two auxiliary components are substituted out, the way the merged
    BM25S dispatch substituted that paper's bespoke tokenizer and benchmark:

      * The learned neural encoder (a masked-language-modeling head emitting
        per-token vocabulary logits -- the heavy ``transformers``/``torch``
        stack) is an *injectable* ``encoder`` rather than a hard dependency.
        A reference learned encoder is provided as ``SpladeEncoder`` (lazy
        import). When no encoder is given, a parameter-free *corpus
        co-occurrence* proxy approximates the expansion signal using only the
        ``sparsetext`` extra's numpy/scipy/sklearn stack, so the scorer is
        usable with no extra deps.
      * The paper's multimodal-Bernoulli *training* procedure (image+text
        contrastive training of the expansion head) is out of scope: this repo
        has no trainer or image call site, so only the inference-time scorer is
        ported.

    Parameters
    ----------
    stringifier
        An ``AttrExpr`` returning one string per chunk; determines how chunks
        are represented as strings for scoring.
    query
        Text string that chunks are scored against.
    encoder
        Optional injectable learned encoder implementing
        ``encode(texts: list[str]) -> scipy.sparse.csr_matrix`` of shape
        ``(len(texts), vocab)``, returning non-negative sparse lexical weights
        per text (already pooled/activated). When omitted, the parameter-free
        corpus co-occurrence proxy is used.
    max_terms
        Expansion-control budget: keep at most this many highest-weight terms
        per text (query and documents). ``None`` keeps all. Defaults to None.
    expand_strength
        Weight on the co-occurrence expansion term in the default proxy
        (ignored when ``encoder`` is given). 0 disables expansion, collapsing
        to an idf-weighted bag of literal terms. Defaults to 1.0.
    kwargs
        ``CountVectorizer`` keyword arguments controlling tokenization (used by
        the default proxy only).
    """

    def __init__(
        self,
        stringifier: AttrExpr,
        query: str,
        encoder=None,
        *,
        max_terms=None,
        expand_strength=1.0,
        **kwargs,
    ):
        if max_terms is not None and (
            not isinstance(max_terms, int)
            or isinstance(max_terms, bool)
            or max_terms < 1
        ):
            raise ValueError(
                f"`max_terms` must be a positive int or None, got {max_terms}"
            )
        if expand_strength < 0:
            raise ValueError(
                f"`expand_strength` must be non-negative, got {expand_strength}"
            )

        self.stringifier = stringifier
        self.query = query
        self.encoder = encoder
        self.max_terms = max_terms
        self.expand_strength = expand_strength
        self.vectorizer = CountVectorizer(**kwargs)

    def __call__(self, chunks: Chunks) -> pa.Array:
        # (The stringifier is responsible for returning its strings
        # in the correct order for the input chunks.)
        strings = self.stringifier(chunks).to_pylist()
        n_chunks = len(strings)
        if n_chunks == 0:
            return pa.array([], type=pa.float64())

        if self.encoder is not None:
            doc_vecs = self.encoder.encode(strings)
            query_vec = self.encoder.encode([self.query])
        else:
            doc_vecs, query_vec = self._proxy_encode(strings)

        # No observable terms (empty corpus / all-stopword strings) -> zero
        # scores for every chunk.
        if doc_vecs.shape[0] == 0 or doc_vecs.shape[1] == 0:
            return pa.array([0.0] * n_chunks, type=pa.float64())

        # Expansion control (shared by both encoder and proxy paths), then a
        # single sparse matvec to score every chunk against the query.
        doc_vecs = _topk_per_row(doc_vecs, self.max_terms)
        query_vec = _topk_per_row(query_vec, self.max_terms)

        scores = (doc_vecs @ query_vec.transpose()).toarray().ravel()
        return pa.array(scores)

    def _proxy_encode(self, strings):
        """Parameter-free learned-sparse proxy.

        Builds an idf-weighted bag of the literal terms, expands it by corpus
        term-term co-occurrence (a distributional stand-in for the learned
        expansion head), and SPLADE-activates the result. Returns
        ``(doc_vecs, query_vec)`` as non-negative CSR matrices, with the
        vectorizer fitted on the documents and the query transformed against
        that same vocabulary + co-occurrence.

        Note: the co-occurrence matrix is ``vocab x vocab`` and is built from
        the supplied chunks, so this default is intended for chunk-level
        scoring rather than web-scale corpora -- pass a learned ``encoder=``
        for large vocabularies.
        """
        tf = self.vectorizer.fit_transform(strings).tocsr()
        n_docs, n_terms = tf.shape
        if tf.nnz == 0 or n_terms == 0:
            return sparse.csr_matrix((n_docs, n_terms)), sparse.csr_matrix((1, n_terms))

        # Lucene-style non-negative idf per term, estimated from the documents.
        df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = np.log(1.0 + n_docs / (df + 0.5))
        idf_diag = sparse.diags(idf)

        # idf-weighted literal term vectors for documents and the query.
        literal = tf.dot(idf_diag).tocsr()
        q_literal = (
            self.vectorizer.transform([self.query]).tocsr().dot(idf_diag).tocsr()
        )

        # Term-term idf-weighted co-occurrence: distributional relatedness proxy
        # for "which vocabulary terms does this text activate".
        cooc = (literal.transpose() @ literal).tocsr()
        cooc.setdiag(0.0)
        cooc.eliminate_zeros()

        def expand(matrix):
            if self.expand_strength == 0:
                return matrix
            return (matrix + self.expand_strength * matrix.dot(cooc)).tocsr()

        return _activate(expand(literal)), _activate(expand(q_literal))


class SpladeEncoder:
    """
    Reference *learned* encoder for ``Splade``: produces SPLADE sparse lexical
    weights from a masked-language-modeling head's logits -- the neural
    term-weighting the default proxy only approximates.

    The heavy ``transformers`` and ``torch`` dependencies are imported lazily
    -- only when this encoder is first used -- so that importing
    ``retrievall.sparsetext`` (and using ``Splade`` with its default proxy)
    never requires them. Install them only if you instantiate this encoder::

        pip install transformers torch

    Pooling follows the SPLADE recipe: per-token vocabulary logits are masked
    to non-pad positions, passed through ``log(1 + ReLU(.))``, and max-pooled
    over the token axis, yielding one non-negative sparse vocabulary vector per
    text. This is the injectable learned encoder the paper trains; here it is
    used at inference only (training is out of scope for this repo).

    Parameters
    ----------
    model_name
        HuggingFace checkpoint of a SPLADE-style masked-LM. Defaults to
        ``"naver/splade-cocondenser-ensembledistil"``.
    device
        Torch device for inference. Defaults to ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        *,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self):
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "SpladeEncoder requires `transformers` and `torch`. Install them "
                "(pip install transformers torch), or omit `encoder=` to use "
                "Splade's dependency-free corpus-co-occurrence proxy."
            ) from exc
        import torch

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(
            self.device
        )
        self._model.eval()

    def encode(self, texts):
        if self._model is None:
            self._load()
        torch = self._torch
        batch = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = self._model(**batch).logits  # (batch, seq, vocab)
        mask = batch["attention_mask"].unsqueeze(-1).to(logits.dtype)
        # SPLADE pooling: zero pad positions, activate, max over token axis.
        weights = torch.log1p(torch.relu(logits * mask)).max(dim=1).values
        return sparse.csr_matrix(weights.cpu().numpy())
