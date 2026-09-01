"""Zero-shot tagging.

This module used to be seventeen strings and a call to `embedder.rank_labels`.
It is now split in two:

* `tag_from_vector` / `tags_from_scores` — **pure arithmetic**. Given a vector
  that is already computed, they do dot products and comparisons. No model, no
  file, no socket. That is what lets the whole library be re-tagged from the
  blobs already in `embeddings.vector` without reading one byte off the disk
  that holds the originals (`scripts/retag.py`).
* `auto_tags` — the original signature, kept because `cli/main.py:174` and
  `core/jobs.py:371`/`:509` call it. It still needs a model, because it starts
  from an image path.

`from core.embedder import ClipEmbedder` used to sit at module scope here, for
one type annotation. It pulled torch — 1.1 GB and several seconds — into every
importer of this file, including paths that only wanted the label list. The
annotation is now a string under `TYPE_CHECKING`;
`tests/test_vocabulary.py::TestTaggerImportsNoTorch` pins it in a fresh
subprocess.

Calibration — and its honest limit
----------------------------------
CLIP's contrastive objective calibrates scores *within* an image (the softmax
that trained it is over the candidate texts for one image) and not *across*
images. Measured on this library, every raw cosine — good match or bad — lands
between 0.158 and 0.278. There is no absolute threshold to pick: 0.24 means
"strong" for one photo and "nothing here" for another, and a threshold tuned for
17 labels does not survive 345.

So thresholds are learned from the corpus, per label: keep label L on photo P
only when `score(P,L)` sits in the upper tail of L's *own* distribution over the
whole library. `calibrate()` records two percentiles per label and
`tags_from_scores` maps them to 0 and 1, so a calibrated score of 1.2 means
"further into this label's tail than 98.5% of the library" regardless of where
that label's raw scores happen to sit.

This is a workaround, not a fix, and it changes what a tag asserts. A tag stops
meaning "this photo depicts a beach" and starts meaning "this photo is unusually
beach-like *for this library*" — so per-label coverage is structurally bounded
by `100 - low_percentile` (with the defaults, no label can reach more than ~10%
of the corpus, which is precisely what stops another `travel` at 81%), and a
library that genuinely is 30% portraits will have portraits it does not tag.
Scores remain incomparable across images; nothing here makes them comparable.
SigLIP's sigmoid loss is what actually fixes this — it trains each
image-text pair independently, so its scores *are* absolute and a fixed
threshold finally means something. Until that lands, this is the best available
and it should not be described as more.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from core.vocabulary import LEGACY_LABELS

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from core.embedder import ClipEmbedder

# Kept for `scripts/reembed.py:784` and anything else importing it by name.
DEFAULT_LABELS: List[str] = list(LEGACY_LABELS)

# A photo gets between MIN and MAX tags — the point being that it does not get
# a constant. Today every tagged photo in production has exactly 6.0.
MIN_TAGS_PER_PHOTO = 3
MAX_TAGS_PER_PHOTO = 8

# Upper tail of a label's own corpus distribution. LOW is the floor for a tag to
# be considered at all; HIGH is where a tag is kept on its own merit. With 345
# labels, HIGH=98.5 yields ~5 strong tags per photo on average, which is the
# band worth aiming at.
LOW_PERCENTILE = 90.0
HIGH_PERCENTILE = 98.5

# A label whose corpus distribution is flat carries no information; without this
# guard its calibrated score would be 0 for every photo, making it weakly
# eligible everywhere — the failure mode this whole module exists to remove.
_DEGENERATE_SPREAD = 1e-6


@dataclass(frozen=True)
class LabelCalibration:
    """Two per-label thresholds, learned from the corpus.

    `low[j]` and `high[j]` are the LOW/HIGH percentiles of label j's scores over
    every photo. They are raw-cosine values, so they are only meaningful next to
    the label matrix and model they were measured with.
    """

    labels: Tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    low_percentile: float
    high_percentile: float
    n_photos: int

    def __len__(self) -> int:
        return len(self.labels)


def calibrate(
    score_matrix: np.ndarray,
    labels: Sequence[str],
    low_percentile: float = LOW_PERCENTILE,
    high_percentile: float = HIGH_PERCENTILE,
) -> LabelCalibration:
    """Learn per-label thresholds from an (n_photos x n_labels) score matrix.

    Pure: this is `np.percentile` down each column. It needs the whole corpus at
    once, which is exactly the thing a per-photo tagging job cannot have and a
    bulk re-tag trivially does.
    """
    scores = np.asarray(score_matrix, dtype="float64")
    if scores.ndim != 2:
        raise ValueError(f"score_matrix must be 2-D, got shape {scores.shape}")
    if scores.shape[1] != len(labels):
        raise ValueError(
            f"score_matrix has {scores.shape[1]} columns but {len(labels)} labels"
        )
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError(
            f"need 0 <= low < high <= 100, got {low_percentile}/{high_percentile}"
        )
    if scores.shape[0] == 0:
        raise ValueError("cannot calibrate against an empty corpus")

    low = np.percentile(scores, low_percentile, axis=0)
    high = np.percentile(scores, high_percentile, axis=0)
    return LabelCalibration(
        labels=tuple(labels),
        low=low.astype("float64"),
        high=high.astype("float64"),
        low_percentile=float(low_percentile),
        high_percentile=float(high_percentile),
        n_photos=int(scores.shape[0]),
    )


def calibrated_scores(scores: np.ndarray,
                      calibration: LabelCalibration) -> np.ndarray:
    """Map each label's LOW percentile to 0 and its HIGH percentile to 1.

    An affine transform per label, so it does not reorder photos *within* a
    label — but it does reorder labels within a photo, which is the whole point.
    Degenerate labels are pushed to -inf rather than 0.
    """
    spread = calibration.high - calibration.low
    dead = spread < _DEGENERATE_SPREAD
    safe = np.where(dead, 1.0, spread)
    out = (np.asarray(scores, dtype="float64") - calibration.low) / safe
    return np.where(dead, -np.inf, out)


def tags_from_scores(
    scores: np.ndarray,
    labels: Sequence[str],
    calibration: Optional[LabelCalibration] = None,
    min_tags: int = MIN_TAGS_PER_PHOTO,
    max_tags: int = MAX_TAGS_PER_PHOTO,
) -> List[Tuple[str, float]]:
    """Pick this photo's tags from its row of raw label scores.

    Returns `(label, raw_cosine)` pairs. With a calibration the list is ordered
    by *corpus-relative* strength, not by raw cosine — deliberately, because raw
    cosine across labels is the quantity this module exists to stop trusting.
    Without one it degrades to plain top-k, which is what `auto_tags` still does.
    """
    scores = np.asarray(scores, dtype="float64").ravel()
    if scores.shape[0] != len(labels):
        raise ValueError(f"{scores.shape[0]} scores but {len(labels)} labels")
    if max_tags <= 0:
        return []

    if calibration is None:
        order = np.argsort(-scores, kind="stable")[:max_tags]
        return [(labels[i], float(scores[i])) for i in order]

    if len(calibration) != len(labels):
        raise ValueError(
            f"calibration covers {len(calibration)} labels, not {len(labels)}"
        )

    cal = calibrated_scores(scores, calibration)
    order = np.argsort(-cal, kind="stable")

    kept = [int(i) for i in order if cal[i] >= 1.0][:max_tags]
    if len(kept) < min_tags:
        # Backfill only from the band between the two percentiles. A photo that
        # is below every label's LOW threshold is genuinely unlike everything in
        # the vocabulary and gets no tags — top-k's inability to say that is the
        # bug being fixed.
        strong = set(kept)
        weak = [int(i) for i in order if i not in strong and 0.0 <= cal[i] < 1.0]
        kept.extend(weak[: min_tags - len(kept)])

    return [(labels[i], float(scores[i])) for i in kept]


def tag_from_vector(
    vector: np.ndarray,
    label_matrix: np.ndarray,
    labels: Sequence[str],
    calibration: Optional[LabelCalibration] = None,
    min_tags: int = MIN_TAGS_PER_PHOTO,
    max_tags: int = MAX_TAGS_PER_PHOTO,
) -> List[Tuple[str, float]]:
    """Tag one photo from a vector that already exists.

    `vector` is an L2-normalised image embedding — which is exactly what
    `embeddings.vector` holds (measured: norm 1.000000 on all 1,910 rows) — and
    `label_matrix` is (n_labels x dim) of L2-normalised text embeddings, so
    `label_matrix @ vector` *is* the vector of cosine similarities. No model is
    involved and no file is read; re-tagging the whole library is one
    `N x dim @ dim x n_labels` GEMM over data already in SQLite.
    """
    vec = np.asarray(vector, dtype="float64").ravel()
    mat = np.asarray(label_matrix, dtype="float64")
    if mat.ndim != 2:
        raise ValueError(f"label_matrix must be 2-D, got shape {mat.shape}")
    if mat.shape[1] != vec.shape[0]:
        raise ValueError(
            f"label_matrix has dim {mat.shape[1]} but the vector has dim {vec.shape[0]}"
        )
    if mat.shape[0] != len(labels):
        raise ValueError(f"{mat.shape[0]} label vectors but {len(labels)} labels")
    return tags_from_scores(mat @ vec, labels, calibration, min_tags, max_tags)


def auto_tags(
    embedder: "ClipEmbedder",
    image_path: str,
    k: int = 6,
) -> List[Tuple[str, float]]:
    """Legacy per-image tagging: load the image, run CLIP, take the top k.

    Kept verbatim in behaviour so `cli/main.py` and the two embedding jobs keep
    working. It is the uncalibrated path — it has one photo and therefore no
    corpus to calibrate against — and it still uses the 17 legacy labels rather
    than the full vocabulary, because embedding 345 prompts per photo inside a
    forked RQ work-horse would cost more than the image pass itself.
    `scripts/retag.py` is what gives a photo its real tags.
    """
    return embedder.rank_labels(image_path, DEFAULT_LABELS, top_k=k)
