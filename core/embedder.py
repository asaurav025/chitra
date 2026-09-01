from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from core.extractor import load_image

#: The model an unset `CHITRA_EMBED_MODEL` means. Every one of the ~1,800 rows
#: in `embeddings` was produced by this, so it stays the default until the
#: re-embed has full coverage.
DEFAULT_EMBED_MODEL = "openai/clip-vit-base-patch32"

SIGLIP2_BASE = "google/siglip2-base-patch16-224"

#: Both embedders pass this to `from_pretrained` rather than inheriting it.
#: transformers is switching the default image processor from the slow PIL
#: implementation to the torchvision-backed fast one; measured on CLIP, that
#: move shifts the output vector by 1.08e-03, which is enough to invalidate a
#: stored corpus while raising nothing. See tests/test_embedder_stability.py.
#: Changing this after a corpus exists means re-embedding that corpus.
USE_FAST_IMAGE_PROCESSOR = False


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


class ClipEmbedder:
    """
    Wrapper around CLIP for image/text embeddings.
    Provides:
      - image_embedding(path)
      - text_embedding(text)
      - rank_labels(image_path, labels, top_k)
    """

    #: Output dimensionality. A class attribute so `/health` and the schema
    #: checks can ask without paying 1.2 GB to load the model first.
    DIM = 512

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        # use_fast=False is deliberate and load-bearing. transformers is moving
        # `from_pretrained` to return the torchvision-backed *fast* image
        # processor by default; it still returns the slow one for this
        # checkpoint only because a slow config was saved alongside the
        # weights, and it warns that this will stop. Measured on the synthetic
        # reference in tests/test_embedder_stability.py, the fast processor
        # moves the output vector by 1.08e-03 — 10x that test's tolerance —
        # because it resizes and rescales differently.
        #
        # Every one of the ~1,800 vectors in `embeddings` was computed with the
        # slow path. Inheriting a default that flips would recompute new ones in
        # a subtly different space, with no error anywhere and search quietly
        # getting worse. Pinning costs nothing: the kwarg has been accepted
        # since at least 4.35.0.
        #
        # Moving to the fast processor is a re-embed decision, not a speed
        # tweak — it needs the whole corpus recomputed in the same pass.
        self.processor = CLIPProcessor.from_pretrained(
            model_name, use_fast=USE_FAST_IMAGE_PROCESSOR
        )
        self.model_name = model_name
        self.dim = self.DIM

    # ------------------------------------------------------------
    # IMAGE → EMBEDDING
    # ------------------------------------------------------------
    def image_embedding(self, file_path: str) -> np.ndarray:
        img = load_image(Path(file_path))

        inputs = self.processor(
            images=img,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        v = feats[0].cpu().numpy().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------
    # TEXT → EMBEDDING
    # ------------------------------------------------------------
    def text_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        v = feats[0].cpu().numpy().astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    # ------------------------------------------------------------
    # LABEL RANKING (used by auto_tags)
    # ------------------------------------------------------------
    def rank_labels(
        self,
        image_path: str,
        labels: list[str],
        top_k: int = 6,
    ):
        """
        Score a list of text labels against an image.
        Returns: list of (label, score) pairs sorted by score desc.
        """
        if not labels:
            return []

        # 1) Image → embedding
        image_vec = self.image_embedding(image_path)  # already L2-normalized

        # 2) Embed all labels at once
        inputs = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        text_vecs = feats.cpu().numpy().astype("float32")

        # Normalize text embeddings
        text_vecs = text_vecs / (np.linalg.norm(text_vecs, axis=1, keepdims=True) + 1e-9)

        # Cosine similarity (image_vec is already normalized)
        sims = text_vecs @ image_vec

        # Sort and pick top-k
        idx = sims.argsort()[::-1][:top_k]

        return [(labels[i], float(sims[i])) for i in idx]


class SiglipEmbedder:
    """SigLIP 2 (`google/siglip2-base-patch16-224`), 768-d.

    Deliberately the same surface as `ClipEmbedder` — `image_embedding`,
    `text_embedding`, `rank_labels` — so `embed_service.py`, the sync client the
    RQ jobs use, and `core/tagger.py` need no change to run against it. The
    migration is meant to be an environment variable, not a refactor.

    **Full `SiglipModel`, fp32, text tower included.** That configuration was
    chosen from measurement, not preference — see
    `docs/plans/siglip2-footprint.md`:

    * **Not vision-only.** `/api/search/photos` embeds arbitrary user text, so
      dropping the text tower would delete the product's primary feature. It is
      also nearly free to keep: with buffalo_l co-resident, the full model peaks
      at 2,137 MB against 2,018 MB for the CLIP it replaces — **+119 MB**.
    * **Not fp16.** Measured, fp16 loses on both axes on this box: peak RSS goes
      *up*, because converting dtype materialises both representations and glibc
      keeps the arena (post-hoc `.half()` on the text tower hits 3,125 MB, 1.7x
      the fp32 model it was supposed to shrink), and the forward pass is 5.6x
      slower because this CPU has no fp16 compute path.

    One operational note that is not obvious from the RSS on day one: the
    `[256000, 768]` multilingual token embedding is 786 MB and is *memory-mapped*
    rather than copied, so only the token rows real queries touch become
    resident. The footprint therefore grows from ~2,137 MB toward ~3,240 MB as
    the query vocabulary widens, with no code change to account for it. Size for
    3,240 MB. The pages are clean and file-backed, so eviction re-reads from
    `~/.cache/huggingface` on the NVMe — never from the failing `/dev/sda`.
    """

    DIM = 768

    #: SigLIP was trained with every text sequence padded to the full 64-token
    #: context, not to the longest item in the batch — otherwise a label's score
    #: would depend on which other labels happened to be ranked beside it.
    #:
    #: `max_length` is passed explicitly even though `SiglipProcessor` already
    #: supplies 64 from its own config. Without it the tokenizer logs "Asking to
    #: pad to max_length but no maximum length is provided ... Default to no
    #: padding", which is wrong — the processor did provide it, and the ids come
    #: out (n, 64) either way (verified: cosine 1.0000001 between the two, and
    #: identical whether a string is embedded alone or in a batch). Being
    #: explicit silences a warning that would otherwise invite someone to
    #: "fix" correct code.
    TEXT_PADDING = "max_length"

    def __init__(self, model_name: str = SIGLIP2_BASE):
        from transformers import AutoProcessor, SiglipModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # No dtype argument: fp32 is the default, and spelling it would mean
        # choosing between `torch_dtype=` and `dtype=`, which changed name
        # inside the >=4.50,<5 range this project supports.
        self.model = SiglipModel.from_pretrained(model_name).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name, use_fast=USE_FAST_IMAGE_PROCESSOR
        )
        self.model_name = model_name
        self.dim = self.DIM
        self.text_max_length = int(self.model.config.text_config.max_position_embeddings)

    def image_embedding(self, file_path: str) -> np.ndarray:
        img = load_image(Path(file_path))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        return _l2(feats[0].cpu().numpy().astype("float32"))

    def text_embedding(self, text: str) -> np.ndarray:
        return self._text_matrix([text])[0]

    def _text_matrix(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of strings into L2-normalised rows."""
        inputs = self.processor(
            text=texts,
            padding=self.TEXT_PADDING,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(input_ids=inputs["input_ids"])

        vecs = feats.cpu().numpy().astype("float32")
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    def rank_labels(self, image_path: str, labels: list[str], top_k: int = 6):
        """Score `labels` against the image. Returns (label, score) desc.

        A note for whoever tunes thresholds on top of this: SigLIP's sigmoid
        loss makes these scores comparable *across* images in a way CLIP's
        softmax contrastive scores are not. That is what lets an absolute
        threshold finally mean something and retires the corpus-relative
        calibration in `core/tagger.py` — but only once the library is
        re-embedded under SigLIP. Until then the two score scales must not be
        mixed.
        """
        if not labels:
            return []

        image_vec = self.image_embedding(image_path)
        sims = self._text_matrix(labels) @ image_vec
        idx = sims.argsort()[::-1][:top_k]
        return [(labels[i], float(sims[i])) for i in idx]


#: Model identifier -> embedder class. Explicit rather than a prefix match, so
#: a typo raises instead of silently resolving to the wrong architecture and
#: writing rows whose `model` column lies about how they were produced.
_EMBEDDERS = {
    "openai/clip-vit-base-patch32": lambda n: ClipEmbedder(n),
    SIGLIP2_BASE: lambda n: SiglipEmbedder(n),
}


def build_embedder(model_name: str | None = None):
    """Construct the embedder named by `model_name`, or `CHITRA_EMBED_MODEL`.

    `CHITRA_EMBED_MODEL` says what the sidecar *computes with*. It is
    deliberately a different variable from `CHITRA_ACTIVE_EMBED_MODEL`, which
    says what `search_photos` *ranks from* (`core.db_async.active_embed_model`).

    Keeping them separate is the entire migration story. The re-embed cannot be
    atomic — ~1,800 photos have to be converted one at a time — so there has to
    be a window where the sidecar already computes SigLIP 768-d vectors while
    search still answers from the complete set of CLIP 512-d rows. Flip
    `CHITRA_EMBED_MODEL` first, re-embed, verify coverage, then flip
    `CHITRA_ACTIVE_EMBED_MODEL`. Rollback is flipping the second one back.

    An unknown identifier raises rather than defaulting: falling back to CLIP
    would write 512-d rows tagged with a name claiming they are something else,
    which is the one error the `model` column exists to make impossible.
    """
    name = model_name or os.environ.get("CHITRA_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    try:
        factory = _EMBEDDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown embedding model {name!r}; known: {sorted(_EMBEDDERS)}"
        ) from None
    return factory(name)
