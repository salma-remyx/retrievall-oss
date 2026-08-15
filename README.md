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

### Context expansion after retrieval
A top-k cut is where a number gets separated from the header that gives it meaning — a table row from its unit header, a figure from the fiscal year it belongs to. `ContextExpand` repairs that cut *after* retrieval by walking the corpus' own structure: it re-cuts each chunk to span the `before` preceding and `after` following chunks of an existing structural chunking, in reading order.
```python
from retrievall.chunkers import FixedSizeChunk
from retrievall.exprs import SimpleStringify
from retrievall.context_expand import ContextExpand
from retrievall.sparsetext.bm25 import BM25
from retrievall.filters import TopK

# Rank rolling chunks by BM25, then re-attach the line above and below each
# hit so a number arrives with its header.
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
    .filter(ContextExpand("line", before=1, after=1))
    .select(text=SimpleStringify())
)
```
`TopK` and `Threshold` also accept a `context=` kwarg that applies the same expansion for you:
```python
corpus.chunk("line").filter(TopK("bm25", 3, context="line", before=1, after=1))
```
Adapted from *Beyond Top-K: Replacing Black-Box Retrieval with Interpretable Agentic Operations* (READ, 2026), which replaces top-k similarity search with structural navigation and bounded span reads over the raw document. Here those operations are a deterministic `ChunkFilter` over the chunk graph the corpus already maintains, rather than agent-driven tools.

## Further reading
See the [`nbs`](/nbs/) directory for more in-depth documentation and examples.

## Contributing
We welcome contributions to Retrievall! Please see our contributing guidelines for details on how to open issues, submit pull requests, and get involved.

## License
Retrievall is open-source software licensed under the MIT license.

## Support
For questions, feature requests or bug reports, please open an issue on this Github repo. 
