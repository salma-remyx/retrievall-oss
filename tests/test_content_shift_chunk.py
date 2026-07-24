import pytest

from retrievall import Chunks, Corpus
from retrievall.content_shift_chunk import ContentShiftChunk
from retrievall.exprs import SimpleStringify

ATOM_WORDS = [
    "The",
    "(quick)",
    "[brown]",
    "fox",
    "jumps!",
    "Over",
    "the",
    "<lazy>",
    "dog",
    "The",
    "~groovy",
    "minute!",
    "dog",
    "bounds",
    "UPON",
    "the",
    "sleepy",
    "fox",
]


class TestContentShiftChunk:
    def test_chunker_drives_corpus_chunk(self, ocr_corpus):
        # The ChunkExpr flows through the *existing* Corpus.chunk() call
        # site in retrievall.core — the same path FixedSizeChunk and
        # RegexMatchChunk use — proving integration rather than a self-test.
        corpus = ocr_corpus

        # Deterministic stub LLM: flags the 3rd passage (index 2) of each
        # 3-word group.
        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        assert isinstance(chunks, Chunks)
        # 18 one-word atoms, 3-word groups, shift at index 2 -> 9 chunks of
        # 2 atoms each.
        assert len(chunks) == 9

    def test_segments_partition_all_atoms(self, ocr_corpus):
        corpus = ocr_corpus

        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        # Every atom of the document appears in exactly one chunk (a
        # partition: no gaps, no overlaps between segments).
        all_atoms = set(corpus.atoms.column("id").to_pylist())
        assigned = chunks.chunk_atoms.column("atom").to_pylist()
        assert set(assigned) == all_atoms
        assert len(assigned) == len(all_atoms)  # no duplicates

    def test_materialized_text_matches_expected_segments(self, ocr_corpus):
        corpus = ocr_corpus

        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))
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

    def test_llm_receives_id_prefixed_passages(self, ocr_corpus):
        corpus = ocr_corpus
        captured: list = []

        def llm(prompt: str) -> str:
            captured.append(prompt)
            return "Answer: ID None"  # no shift -> consume the whole group

        corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        # The first call presents exactly 3 ID-prefixed passages and asks for
        # the paper's 'Answer: ID XXXX' contract.
        first = captured[0]
        assert first.count("ID 0000:") == 1
        assert first.count("ID 0001:") == 1
        assert first.count("ID 0002:") == 1
        assert "ID 0003:" not in first
        assert "'Answer: ID XXXX'" in first

    def test_word_budget_groups_passages_by_words(self, ocr_corpus):
        # Groups are formed by a word budget, not a fixed passage count.
        corpus = ocr_corpus

        def llm(prompt: str) -> str:
            return "Answer: ID None"  # no shift -> the whole group is a chunk

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=2))

        # 18 one-word atoms / 2 words per group -> 9 uniform chunks.
        assert len(chunks) == 9

    def test_overbudget_passage_emitted_without_querying_llm(self, ocr_corpus):
        corpus = ocr_corpus
        calls: list = []

        def llm(prompt: str) -> int:
            calls.append(prompt)
            return 1

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=1))

        # Every 1-word atom fills the budget alone -> single-atom chunks, and
        # the LLM is never shown a group it could not split.
        assert len(chunks) == 18
        assert calls == []

    def test_scan_resumes_from_split_point(self, ocr_corpus):
        # LumberChunker resumes scanning from the flagged passage: no sliding
        # window, no overlap, no re-asking about earlier text.
        corpus = ocr_corpus
        captured: list = []

        def llm(prompt: str) -> str:
            captured.append(prompt)
            return "Answer: ID 0001"  # shift at the 2nd passage of each group

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        # Each call cuts one leading atom and the next group starts at the
        # flagged passage, so prompt k leads with atom k.
        assert len(chunks) == 18
        for k, prompt in enumerate(captured):
            first_passage = prompt.split("Passages:\n", 1)[1].splitlines()[0]
            assert first_passage == f"ID 0000: {ATOM_WORDS[k]}"

    def test_string_llm_output_is_parsed(self, ocr_corpus):
        # A real LLM client returns free text; the identifier is extracted.
        corpus = ocr_corpus

        def llm(prompt: str) -> str:
            return "The content shifts after passage 1."

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=2))

        # shift after passage 1 -> each chunk is a single atom -> 18 chunks.
        assert len(chunks) == 18

    def test_cot_answer_line_is_parsed(self, ocr_corpus):
        # Chain-of-thought output: reasoning with distractor numbers and IDs
        # above the final answer line must not shadow the answer.
        corpus = ocr_corpus

        def llm(prompt: str) -> str:
            return (
                "Passages ID 0000 and ID 0001 both describe the same 3 "
                "scenes, but the tone changes at the next passage.\n"
                "Answer: ID 0002"
            )

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        # flagged index 2 -> 2 leading atoms per chunk -> 9 chunks.
        assert len(chunks) == 9

    def test_no_shift_answer_keeps_group_together(self, ocr_corpus):
        corpus = ocr_corpus

        def llm(prompt: str) -> str:
            return "Answer: ID None"

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        # 18 atoms / 3 words per group -> 6 uniform chunks.
        assert len(chunks) == 6

    def test_invalid_max_group_words_rejected(self):
        def llm(prompt: str) -> int:
            return 1

        with pytest.raises(ValueError):
            ContentShiftChunk("document", llm, max_group_words=0)

    def test_multidoc_segments_independently(self, tesseract_table):
        from retrievall.ocr import corpus_from_tesseract_table

        corp1 = corpus_from_tesseract_table(tesseract_table, document_id="abc123")
        corp2 = corpus_from_tesseract_table(tesseract_table, document_id="edf456")
        corpus = Corpus.merge([corp1, corp2])

        def llm(prompt: str) -> int:
            return 2

        chunks = corpus.chunk(ContentShiftChunk("document", llm, max_group_words=3))

        assert isinstance(chunks, Chunks)
        # Two documents, 9 chunks each -> 18 total.
        assert len(chunks) == 18
