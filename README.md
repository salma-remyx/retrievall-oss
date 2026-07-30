> [!WARNING]  
> This repository was put together for demonstration purposes, and might not be actively updated!

# 🎣 Retrievall
Retrievall ("retrieve all") is a general framework for retrieval tasks. It aims to let you retrieve whatever you want from anything you want however you want to.

## Installation
Retrievall comes with a set of basic components and features. For leanness and extensibility, more specialized components are separated out into extension modules.

Retrievall can be installed from its GitHub source via pip
```
# In a dependency list
retrievall @ git+ssh://git@github.com/CohereHealth/retrievall-oss
# From the command line
pip install git+ssh://git@github.com/CohereHealth/retrievall-oss

```


### Extension modules
Extension module dependencies can be installed using install [extra options](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras):
```
pip install retrievall-oss[sparsetext]

```

The built-in extension module components or functions are accessed as sub-modules of Retrievall, e.g.:
```python
from retrievall.sparsetext import Tfidf
```

The currently available extension modules are:
* `sparsetext`: Sparse text-based retrieval components, like `Tfidf`
* `ocr`: Representations of and interactions with OCR data, like Tesseract.

## Quickstart
> ⚠️ This is a pre-1.0 version, so the public API may change rapidly.

Retrievall is all about retrieving pieces of information (`Chunks`) from a large collection of information or documents (a `Corpus`). The corpus, chunks, and retrieval methods are all user-configurable.

A corpus can be created manually, or helper methods may be used to construct one from existing data:
```python
from retrievall.ocr import corpus_from_tesseract_table

tesseract_ocr_table # pa.Table; Tesseract-style OCR data

corpus = corpus_from_tesseract_table(ocr_table, document_id="abc123")
```

Once a corpus is created, you can chunk, filter, and retrieve from it using any combination of components. The retrieval API is somewhat similar to [Polars'](https://docs.pola.rs/user-guide/getting-started/#expressions) or [Ibis'](https://ibis-project.org/tutorials/getting_started#chaining-it-all-together) chained expressions.

### Examples
```python
from retrievall.exprs import SimpleStringify
from retrievall.filters import Threshold

# Retrieve the text of the first 2 lines (a built-in chunk in this corpus)
# based on their `ordinal` value (a built-in attribute for line chunks)
(
    corpus.chunk("line")
    .filter(Threshold("ordinal", "<=", 2))
    .select(text=SimpleStringify())
) # `.select()` returns a pa.Table
```

```python
from retrievall.chunkers import FixedSizeChunk
from retrievall.exprs import SimpleStringify
from retrievall.sparsetext import Tfidf
from retrievall.filters import TopK

# Create rolling 64-token chunks from within pages, find
# their TF-IDF relevance score for a query, and select the 
# 3 chunks with the highest relevance score.
(
    corpus.chunk(
        FixedSizeChunk("page", 64, offset=-32)
    )
    .enrich(
        tfidf=Tfidf(
            SimpleStringify(),
            query="brown fox"
        )
    )
    .filter(TopK("tfidf", 3))
    .select(text=SimpleStringify())
)
```


### BM25 sparse retrieval
The `sparsetext` module also provides `BM25`, the canonical sibling of TF-IDF for sparse lexical retrieval. It scores chunks against a query with the Okapi BM25 ranking function and the same `.enrich()` / `.select()` contract as `Tfidf`:
```python
from retrievall.chunkers import FixedSizeChunk
from retrievall.exprs import SimpleStringify
from retrievall.sparsetext.bm25 import BM25
from retrievall.filters import TopK

# Rank rolling 64-token page chunks by BM25 relevance to a query.
(
    corpus.chunk(
        FixedSizeChunk("page", 64, offset=-32)
    )
    .enrich(
        bm25=BM25(
            SimpleStringify(),
            query="brown fox"
        )
    )
    .filter(TopK("bm25", 3))
    .select(text=SimpleStringify())
)
```

## Further reading
See the [`nbs`](/nbs/) directory for more in-depth documentation and examples.

## Contributing
We welcome contributions to Retrievall! Please see our contributing guidelines for details on how to open issues, submit pull requests, and get involved.

## License
Retrievall is open-source software licensed under the MIT license.

## Support
For questions, feature requests or bug reports, please open an issue on this Github repo.

### Reciprocal rank fusion (hybrid retrieval)
The `sparsetext` module also provides `ReciprocalRankFusion`, which combines several retrieval channels into a single relevance score per chunk using Reciprocal Rank Fusion (RRF). Because RRF fuses *ranks* rather than raw scores, it mixes channels whose score scales are incomparable — sparse lexical scorers like `Tfidf`/`BM25` alongside any future dense or knowledge-graph scorer — without calibrating them. It has the same `.enrich()` / `.select()` contract as the individual scorers:
```python
from retrievall.exprs import SimpleStringify
from retrievall.sparsetext import BM25, Tfidf
from retrievall.sparsetext.fusion import ReciprocalRankFusion
from retrievall.filters import TopK

# Fuse TF-IDF and BM25 into one ranking for a query, then keep the top chunks.
(
    corpus.chunk("page")
    .enrich(
        fused=ReciprocalRankFusion(
            [
                Tfidf(SimpleStringify(), query="brown fox"),
                BM25(SimpleStringify(), query="brown fox"),
            ],
            # Optional per-channel weights (normalized to sum to 1); default is
            # a uniform fusion. Up-weight the channel you trust for this query.
            weights=[0.4, 0.6],
        )
    )
    .filter(TopK("fused", 3))
    .select(text=SimpleStringify())
)
```
The fusion core is adapted from APS-RAG ("A corrective agentic hybrid RAG and an operations-grounded evaluation for a scientific facility", [arXiv:2607.24663](https://arxiv.org/abs/2607.24663)), whose retrieval engine fuses dense, sparse, and knowledge-graph channels with query-type-adaptive reciprocal-rank fusion; here the per-channel `weights` expose that adaptive knob.
