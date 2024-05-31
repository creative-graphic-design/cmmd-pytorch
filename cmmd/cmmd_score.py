import os
from typing import Optional

import numpy as np

from cmmd.distance import mmd
from cmmd.embedding import ClipEmbeddingModel
from cmmd.io_util import compute_embeddings_for_dir


def compute_cmmd(
    ref_dir: os.PathLike,
    eval_dir: os.PathLike,
    ref_embed_file: Optional[os.PathLike] = None,
    batch_size: int = 32,
    max_count: int = -1,
) -> float:
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError(
            "`ref_dir` and `ref_embed_file` both cannot be set at the same time."
        )
    embedding_model = ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = compute_embeddings_for_dir(
            ref_dir, embedding_model, batch_size, max_count
        ).astype("float32")
    eval_embs = compute_embeddings_for_dir(
        eval_dir, embedding_model, batch_size, max_count
    ).astype("float32")
    val = mmd(ref_embs, eval_embs)
    return val.item()
