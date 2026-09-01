"""The zero-shot tag vocabulary.

Why this file exists
--------------------
`core/tagger.py` shipped with 17 hardcoded labels and a fixed top-6. Measured on
the production library (1,909 tagged photos, 11,456 tag rows):

    travel      1,554 photos   81.4%
    outdoors      903          47.3%
    portrait      868          45.5%
    ...
    every photo has exactly 6.0 tags

Raw CLIP cosine over all 11k stored scores spans **0.158 to 0.278**, median
0.204, with a 0.03 gap between a photo's best and worst kept label. Below rank
1-2 the ranking carries almost no signal, and top-k hands back k labels whether
any of them fit or not. `travel` is not a fact about the library; it is what
"least unlike this photo, out of 17 strings" returns when the 17 strings do not
cover the space.

The fix is more labels — a few hundred, so the nearest one is actually near —
plus per-label calibration in `core.tagger`. Both halves are needed: 300 labels
under fixed top-6 would be *worse*, because the 6th-best of 300 is even more
arbitrary than the 6th-best of 17.

Grouping
--------
Labels carry a facet so a future `GET /api/tags` can present them as something
other than one flat 300-item list, and so a UI can offer "places" separately
from "occasions". The facet is presentation metadata; it does not affect
scoring.

Prompting
---------
`PROMPT_TEMPLATE = "a photo of {label}"` is applied uniformly, so the stored tag
string stays clean ("beach", not "a photo of a beach"). Some labels read
slightly ungrammatically inside the template ("a photo of portrait"). That is
deliberate and harmless here: a per-label prompt artefact is a *constant offset
on that label's column*, and `core.tagger.calibrate` thresholds every label
against its own corpus distribution — which absorbs exactly that kind of
constant. What it cannot absorb is an *inconsistent* template, which is why the
template is part of `vocab_fingerprint()`.

Versioning
----------
`VOCAB_VERSION` and `vocab_fingerprint()` identify the vocabulary. The
fingerprint names the cached label-vector file, so a label added here can never
be silently scored against a stale matrix, and `tag_source()` stamps
`tags.source` so a bad vocabulary is one `DELETE FROM tags WHERE source = ...`
from reverted.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

VOCAB_VERSION = "v2"

# Part of the fingerprint on purpose: changing this string moves every label
# vector, which invalidates every cached matrix and every stored score.
PROMPT_TEMPLATE = "a photo of {label}"

# The 17 labels that shipped. Every one is retained below, so no row in the
# existing `tags` table names a label the new vocabulary cannot re-score.
LEGACY_LABELS: Tuple[str, ...] = (
    "portrait", "selfie", "group photo", "family", "friends", "landscape",
    "city", "night", "sunset", "food", "pets", "indoors", "outdoors",
    "travel", "wedding", "party", "sports",
)


FACETS: Dict[str, Tuple[str, ...]] = {
    # ------------------------------------------------------------------
    # What kind of place or view the photo *is*.
    # ------------------------------------------------------------------
    "scene": (
        "landscape", "indoors", "outdoors",
        "beach", "ocean", "lake", "river", "waterfall", "pond",
        "mountains", "hills", "valley", "cliff", "canyon", "glacier",
        "forest", "jungle", "woodland", "meadow", "grassland", "desert",
        "farmland", "rice fields", "vineyard", "garden", "park",
        "snowy landscape", "sand dunes", "swamp", "coastline", "harbour",
        "island", "cave", "volcano", "waterside", "countryside",
        "skyline", "cityscape", "aerial view", "panorama",
        "sky", "clouds", "storm", "rainbow", "fog", "sunrise",
        "starry sky", "northern lights", "reflection in water",
        "empty street", "ruins", "construction site",
    ),
    # ------------------------------------------------------------------
    # A named venue or built place, rather than a view of nature.
    # ------------------------------------------------------------------
    "place": (
        "city",
        "home", "living room", "kitchen", "bedroom", "bathroom", "balcony",
        "backyard", "rooftop", "staircase", "hallway", "office", "classroom",
        "library", "hospital", "gym", "swimming pool", "playground",
        "restaurant", "cafe", "bar", "nightclub", "hotel room", "shop",
        "market", "supermarket", "shopping mall", "bakery", "street food stall",
        "museum", "art gallery", "theatre", "cinema", "concert hall", "stadium",
        "temple", "church", "mosque", "monastery", "cemetery",
        "airport", "train station", "bus station", "subway", "parking lot",
        "bridge", "tunnel", "lighthouse", "castle", "monument", "skyscraper",
        "village", "suburb", "campsite", "amusement park", "zoo", "aquarium",
    ),
    # ------------------------------------------------------------------
    # What someone is doing.
    # ------------------------------------------------------------------
    "activity": (
        "travel", "sports",
        "hiking", "camping", "climbing", "cycling", "running", "swimming",
        "surfing", "skiing", "snowboarding", "skating", "sailing", "kayaking",
        "fishing", "diving", "yoga", "dancing", "playing football",
        "playing cricket", "playing basketball", "playing tennis",
        "riding a motorcycle", "driving", "road trip", "flying",
        "walking the dog", "gardening", "cooking", "baking", "eating",
        "drinking coffee", "shopping", "reading", "writing", "studying",
        "working on a laptop", "meeting", "presentation", "interview",
        "playing guitar", "playing piano", "singing", "playing video games",
        "painting", "drawing", "taking a photograph", "fixing something",
        "cleaning", "sleeping", "sunbathing", "protest march", "parade",
    ),
    # ------------------------------------------------------------------
    # A thing that dominates the frame.
    # ------------------------------------------------------------------
    "object": (
        "food", "pets",
        "breakfast", "dessert", "cake", "pizza", "noodles", "curry", "salad",
        "barbecue", "fruit", "vegetables", "coffee", "cocktail", "wine",
        "beer", "ice cream", "street food",
        "dog", "cat", "bird", "horse", "cow", "elephant", "monkey", "fish",
        "butterfly", "insect", "wildlife", "farm animals",
        "flowers", "trees", "plants", "houseplant", "mushroom",
        "car", "motorcycle", "bicycle", "bus", "train", "aeroplane", "boat",
        "helicopter", "truck",
        "book", "laptop", "smartphone", "camera", "watch", "shoes",
        "clothing", "jewellery", "handbag", "furniture", "artwork",
        "sculpture", "graffiti", "poster", "sign", "map", "document",
        "screenshot", "whiteboard", "receipt", "toy", "musical instrument",
        "candles", "balloons", "fireworks", "campfire", "boxes",
    ),
    # ------------------------------------------------------------------
    # Who is in it.
    # ------------------------------------------------------------------
    "people": (
        "portrait", "selfie", "group photo", "family", "friends",
        "couple", "crowd", "child", "children playing", "baby", "toddler",
        "teenager", "elderly person", "man", "woman", "colleagues",
        "team photo", "candid portrait", "headshot", "silhouette of a person",
        "someone from behind", "hands", "feet", "face close-up",
        "person alone", "large gathering", "posed group",
    ),
    # ------------------------------------------------------------------
    # When.
    # ------------------------------------------------------------------
    "time_of_day": (
        "night", "sunset",
        "dawn", "morning", "midday", "afternoon", "golden hour", "dusk",
        "twilight", "after dark", "bright daylight", "overcast day",
    ),
    "season": (
        "spring", "summer", "autumn", "winter",
        "snow", "rain", "monsoon", "heatwave", "autumn leaves",
        "cherry blossom", "frost",
    ),
    # ------------------------------------------------------------------
    # The occasion the photo was taken for.
    # ------------------------------------------------------------------
    "occasion": (
        "wedding", "party",
        "birthday party", "anniversary", "graduation", "baby shower",
        "engagement", "funeral", "religious ceremony", "festival",
        "diwali celebration", "christmas", "new year celebration",
        "holi celebration", "eid celebration", "halloween", "carnival",
        "concert", "live music", "conference", "sporting event", "picnic",
        "dinner party", "housewarming", "reunion", "farewell", "holiday trip",
        "vacation", "day out", "night out",
    ),
    # ------------------------------------------------------------------
    # How it was taken — the facet a photo library actually needs for
    # filtering, and the one 17 labels had nothing for.
    # ------------------------------------------------------------------
    "photo_style": (
        "black and white photo", "sepia photo", "vintage photo", "film photo",
        "close-up", "macro photo", "wide angle shot", "long exposure",
        "motion blur", "blurry photo", "dark photo", "overexposed photo",
        "high contrast photo", "colourful photo", "minimal composition",
        "symmetrical composition", "flat lay", "top-down view",
        "low angle shot", "backlit photo", "bokeh background",
        "drone photo", "underwater photo", "night photography",
        "professional photograph", "amateur snapshot", "scanned photograph",
        "screenshot of a screen", "text document scan", "collage",
        "photo with a caption", "mirror selfie",
    ),
}


LABELS: Tuple[str, ...] = tuple(
    label for labels in FACETS.values() for label in labels
)

_FACET_OF: Dict[str, str] = {
    label: facet for facet, labels in FACETS.items() for label in labels
}

# Loud rather than subtle: a duplicated label would give the label matrix two
# identical rows and the tag table two paths to the same string.
if len(_FACET_OF) != len(LABELS):
    _dupes = sorted({lab for lab in LABELS if LABELS.count(lab) > 1})
    raise RuntimeError(f"duplicate labels in vocabulary: {_dupes}")

_missing_legacy = [lab for lab in LEGACY_LABELS if lab not in _FACET_OF]
if _missing_legacy:
    raise RuntimeError(
        f"legacy labels dropped from the vocabulary: {_missing_legacy} — "
        "every one of them names rows in the existing `tags` table"
    )


def facet_of(label: str) -> str:
    """Which facet a label belongs to. Raises for an unknown label."""
    try:
        return _FACET_OF[label]
    except KeyError:
        raise KeyError(f"{label!r} is not in the vocabulary") from None


def prompt_for(label: str, template: Optional[str] = None) -> str:
    return (template or PROMPT_TEMPLATE).format(label=label)


def prompts(labels: Optional[Sequence[str]] = None,
            template: Optional[str] = None) -> list:
    """The strings actually handed to the text encoder, in label order.

    Order is load-bearing: row *i* of the cached label matrix is
    `LABELS[i]`, and nothing downstream re-checks that by name.
    """
    return [prompt_for(lab, template) for lab in (LABELS if labels is None else labels)]


def vocab_fingerprint(labels: Optional[Iterable[str]] = None,
                      version: Optional[str] = None,
                      template: Optional[str] = None) -> str:
    """A short, stable id for (labels, version, template).

    All three go in. The labels and their order decide what each matrix row
    means; the template decides where every vector *is*. A cached matrix built
    under a different template is not stale-looking, it is simply wrong, and
    the numbers give no hint of it — so the filename has to carry the
    distinction.
    """
    labs = tuple(LABELS if labels is None else labels)
    h = hashlib.sha256()
    h.update((version or VOCAB_VERSION).encode("utf-8"))
    h.update(b"\x00")
    h.update((template or PROMPT_TEMPLATE).encode("utf-8"))
    h.update(b"\x00")
    for lab in labs:
        h.update(lab.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


# Keeps the existing `clip-vitb32/vocab-v1` shape for the model we actually run;
# anything else gets a sanitised fallback rather than a wrong short name.
_MODEL_SLUGS = {
    "openai/clip-vit-base-patch32": "clip-vitb32",
    "openai/clip-vit-large-patch14": "clip-vitl14",
}


def model_slug(model: str) -> str:
    if model in _MODEL_SLUGS:
        return _MODEL_SLUGS[model]
    tail = (model or "unknown").rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail) or "unknown"


def tag_source(model: str, version: Optional[str] = None) -> str:
    """The value stamped into `tags.source`.

    Provenance, not identity — `tags` is unique on `(photo_id, tag)`. It is what
    makes the whole re-tag revertible with one DELETE, so it must differ from
    `db.DEFAULT_TAG_SOURCE` whenever the vocabulary does.
    """
    return f"{model_slug(model)}/vocab-{version or VOCAB_VERSION}"
