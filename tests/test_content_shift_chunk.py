import pytest

from retrievall import Chunks, Corpus
from retrievall.content_shift_chunk import ContentShiftChunk
from retrievall.exprs import SimpleStringify


class TestContentShiftChunk:
    def test_chunker_drives_corpus_chunk(self, ocr_corpus):
        # The new ChunkExpr flows through the *existing* Corpus.chunk() call
        # site in retrievall.core — the same path FixedSizeChunk and
        # RegexMatchChunk use — proving integration rather than a self-test.
        corpus = ocr_corpus

        # Deterministic stub LLM: always reports a shift after the 2nd passage
        # of each 3-passage group.
        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, group_size=3))

        assert isinstance(chunks, Chunks)
        # 18 atoms, group_size 3, shift after 2 -> 9 chunks of 2 atoms each.
        assert len(chunks) == 9

    def test_segments_partition_all_atoms(self, ocr_corpus):
        corpus = ocr_corpus

        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, group_size=3))

        # Every atom of the document appears in exactly one chunk (a partition).
        all_atoms = set(corpus.atoms.column("id").to_pylist())
        assigned = chunks.chunk_atoms.column("atom").to_pylist()
        assert set(assigned) == all_atoms
        assert len(assigned) == len(all_atoms)  # no duplicates

    def test_materialized_text_matches_expected_segments(self, ocr_corpus):
        corpus = ocr_corpus

        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, group_size=3))
        res = chunks.select(text=SimpleStringify(delimiter=" "))

        assert res["text"].to_pylist() == [
            "The (quick)",
            "[brown] fox",
            "jumps! Over",
            "the <lazy>",
            "dog The",
            "~groovy minute!",
            "dog bounds",
            "UPON the",
            "sleepy fox",
        ]

    def test_llm_receives_numbered_passages(self, ocr_corpus):
        corpus = ocr_corpus
        captured: list = []

        def llm(prompt: str) -> int:
            captured.append(prompt)
            return 3  # no shift within the group -> consume the whole group

        corpus.chunk(ContentShiftChunk("document", llm, group_size=3))

        # The first call should present exactly 3 numbered passages.
        first = captured[0]
        assert first.count("[1]") == 1
        assert first.count("[2]") == 1
        assert first.count("[3]") == 1
        assert "1 and 3" in first

    def test_string_llm_output_is_parsed(self, ocr_corpus):
        # A real LLM client returns free text; the integer is extracted from it.
        corpus = ocr_corpus

        def llm(prompt: str) -> str:
            return "The content shifts after passage 1."

        chunks = corpus.chunk(ContentShiftChunk("document", llm, group_size=2))

        # shift after passage 1 -> each chunk is a single atom -> 18 chunks.
        assert len(chunks) == 18

    def test_no_shift_yields_uniform_chunks(self, ocr_corpus):
        corpus = ocr_corpus

        def llm(prompt: str) -> int:
            # group_size -> no shift detected -> the whole group is one chunk.
            return 3

        chunks = corpus.chunk(ContentShiftChunk("document", llm, group_size=3))

        # 18 atoms / 3 per group -> 6 uniform chunks.
        assert len(chunks) == 6

    def test_invalid_group_size_rejected(self):
        def llm(prompt: str) -> int:
            return 1

        with pytest.raises(ValueError):
            ContentShiftChunk("document", llm, group_size=0)

    def test_multidoc_segments_independently(self, tesseract_table):
        from retrievall.ocr import corpus_from_tesseract_table

        corp1 = corpus_from_tesseract_table(tesseract_table, document_id="abc123")
        corp2 = corpus_from_tesseract_table(tesseract_table, document_id="edf456")
        corpus = Corpus.merge([corp1, corp2])

        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, group_size=3))

        assert isinstance(chunks, Chunks)
        # Two documents, 9 chunks each -> 18 total.
        assert len(chunks) == 18
