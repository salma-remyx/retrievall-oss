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
* `dense`: Dense and hybrid retrieval components — `DenseEmbedding` plus `HybridRRF` reciprocal-rank fusion (adapted from UEmbed).

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

### Dense and hybrid retrieval
The `dense` module provides `DenseEmbedding`, the dense counterpart to `Tfidf` / `BM25`: it embeds chunks and a query and scores them by cosine similarity, using the same `.enrich()` / `.select()` contract.

`DenseEmbedding` takes an injectable `embedder` (any callable mapping `list[str] -> (n, d)` vectors). Without one, it lazily loads a [sentence-transformers](https://www.sbert.net/) backend on first use — so install it separately (e.g. `pip install sentence-transformers`) and the heavy model stays out of the slim core and out of import time.

```python
from retrievall.chunkers import FixedSizeChunk
from retrievall.exprs import SimpleStringify
from retrievall.dense import DenseEmbedding
from retrievall.filters import TopK

# Rank rolling 64-token page chunks by dense semantic relevance to a query.
(
    corpus.chunk(
        FixedSizeChunk("page", 64, offset=-32)
    )
    .enrich(
        dense=DenseEmbedding(
            SimpleStringify(),
            query="brown fox",
        )
    )
    .filter(TopK("dense", 3))
    .select(text=SimpleStringify())
)
```

`HybridRRF` fuses two or more scorers into one ranking with reciprocal rank fusion (RRF) — the standard way to blend a sparse lexical signal with a dense semantic signal, the core idea of [UEmbed (Unified Sparse and Dense Multimodal Embeddings)](https://arxiv.org/abs/2608.02583v1). UEmbed produces both representations inside one model; `HybridRRF` does the equivalent fusion at the framework level, combining any scorers (here `BM25` and `DenseEmbedding`) into a single hybrid retriever:

```python
from retrievall.sparsetext import BM25
from retrievall.dense import DenseEmbedding, HybridRRF

(
    corpus.chunk(
        FixedSizeChunk("page", 64, offset=-32)
    )
    .enrich(
        hybrid=HybridRRF(
            BM25(SimpleStringify(), query="brown fox"),
            DenseEmbedding(SimpleStringify(), query="brown fox"),
        )
    )
    .filter(TopK("hybrid", 3))
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
