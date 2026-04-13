"""Memory-mapped reader for browsecomp-plus corpus.jsonl.

Build the index once before use:

    uv run scripts/build_corpus_index.py
"""

import json
import mmap

CORPUS_PATH = "/mnt/data_4tb/datasets/browsecomp-plus-corpus/corpus.jsonl"
INDEX_PATH = "/mnt/data_4tb/datasets/browsecomp-plus-corpus/corpus_index.json"


class CorpusReader:
    """Memory-mapped reader for corpus.jsonl.

    Loads only the byte-offset index (~3 MB) on construction; individual
    documents are fetched by seeking directly into the mmap'd file, so memory
    usage is proportional to actual accesses rather than the full 3.3 GB.

    Safe to instantiate once per process (including worker subprocesses) — the
    OS shares physical pages of the mmap across processes automatically.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH, index_path: str = INDEX_PATH) -> None:
        with open(index_path) as f:
            self._index: dict[str, list[int]] = json.load(f)
        self._f = open(corpus_path, "rb")
        self._mmap = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, docid: str) -> dict:
        offset, length = self._index[str(docid)]
        return json.loads(self._mmap[offset : offset + length])


_corpus_reader: CorpusReader | None = None


def get_corpus_reader() -> CorpusReader:
    global _corpus_reader
    if _corpus_reader is None:
        _corpus_reader = CorpusReader()
    return _corpus_reader
