"""BM25 keyword search over chunks, in memory and without dependencies.

Vectors find texts about the same subject; BM25 finds texts that contain the words and figures
of the question ("Coteccons", "19,6 triệu"), which a vector often blurs. The two are combined
by `reciprocal_rank_fusion`.

Tokens are the syllables Vietnamese is written in (lower-cased, diacritics kept) plus every
pair of neighbouring syllables, so "lãi suất" and "cho vay" count as words and not only as
four loose syllables. A number such as "19,6" or "12.038" stays one token.

This is "BM25 (Okapi)" as defined by Robertson and Walker, computed here; it is not a
Postgres full-text rank, which is a different formula.
"""

import math
import re
from array import array
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise

_TOKEN = re.compile(r"\w+(?:[.,]\w+)*")


def tokenize(text: str) -> list[str]:
    syllables = _TOKEN.findall(text.lower())
    return syllables + [f"{a} {b}" for a, b in pairwise(syllables)]


class Bm25Index:
    def __init__(
        self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1, self.b = k1, b
        self.lengths = array("I")
        # token -> (document ids, term frequencies), parallel arrays to keep millions of
        # postings small.
        self._postings: dict[str, tuple[array[int], array[int]]] = {}
        for doc_id, text in enumerate(documents):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                ids, tfs = self._postings.setdefault(token, (array("I"), array("H")))
                ids.append(doc_id)
                tfs.append(min(frequency, 65535))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def scores(
        self, query: str, allowed: Callable[[int], bool] | None = None
    ) -> dict[int, float]:
        """BM25 score of every document that shares a token with `query` (best are highest)."""
        total = len(self.lengths)
        result: dict[int, float] = {}
        for token in set(tokenize(query)):
            posting = self._postings.get(token)
            if posting is None:
                continue
            ids, tfs = posting
            idf = math.log(1 + (total - len(ids) + 0.5) / (len(ids) + 0.5))
            for doc_id, tf in zip(ids, tfs):
                if allowed is not None and not allowed(doc_id):
                    continue
                norm = tf + self.k1 * (
                    1 - self.b + self.b * self.lengths[doc_id] / self.average_length
                )
                result[doc_id] = (
                    result.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / norm
                )
        return result


def reciprocal_rank_fusion(
    rankings: Sequence[Mapping[int, float]], k: int = 60, depth: int = 100
) -> dict[int, float]:
    """Merge score tables by rank alone: an item scores 1 / (k + rank) in each list it is in.

    Scores of different kinds (cosine similarity, BM25) cannot be added, but ranks can. Only
    the best `depth` of each list count; an item missing from a list adds nothing for it.
    """
    fused: dict[int, float] = {}
    for scores in rankings:
        best = sorted(scores, key=lambda item: scores[item], reverse=True)[:depth]
        for rank, item in enumerate(best, start=1):
            fused[item] = fused.get(item, 0.0) + 1 / (k + rank)
    return fused
