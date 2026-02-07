"""Shared data processing utilities for UniVL dataloaders.

Provides common operations used across all dataset-specific dataloaders:
  - Masked Language Modeling (MLM) token masking
  - Masked Frame Modeling (MFM) video frame masking
  - Token padding helpers

Centralising these eliminates the copy-pasted masking logic that was
previously duplicated in every dataloader file.
"""

import random
import numpy as np


# ---------------------------------------------------------------------------
# Masked Language Modeling (MLM)
# ---------------------------------------------------------------------------


def mask_tokens(words, tokenizer, mlm_probability=0.15):
    """Apply BERT-style Masked Language Model masking to a token sequence.

    Masking strategy (following Devlin et al., 2019):
      - Each non-special token is selected for masking with probability
        ``mlm_probability`` (default 15%).
      - Of the selected tokens:
            80% are replaced with ``[MASK]``
            10% are replaced with a random vocabulary token
            10% are left unchanged
      - The first and last tokens (typically ``[CLS]`` and ``[SEP]``) are
        never masked.

    Args:
        words: List of token strings.  Should already include [CLS] at the
            start and [SEP] at the end.
        tokenizer: Tokenizer instance with a ``vocab`` dict mapping tokens
            to integer ids.
        mlm_probability: Probability of masking each eligible token.

    Returns:
        masked_tokens: Copy of *words* with masking applied.
        token_labels: List of integer labels — the original vocab id for
            masked positions, and ``-1`` for unmasked positions (ignored by
            ``CrossEntropyLoss``).
    """
    token_labels = []
    masked_tokens = words.copy()

    for token_id, token in enumerate(masked_tokens):
        # Never mask the leading [CLS] or trailing [SEP]
        if token_id == 0 or token_id == len(masked_tokens) - 1:
            token_labels.append(-1)
            continue

        prob = random.random()
        if prob < mlm_probability:
            prob /= mlm_probability  # rescale to [0, 1) for sub-decisions

            if prob < 0.8:
                # 80 %: replace with [MASK]
                masked_tokens[token_id] = "[MASK]"
            elif prob < 0.9:
                # 10 %: replace with a random vocabulary token
                masked_tokens[token_id] = random.choice(list(tokenizer.vocab.keys()))
            # remaining 10 %: keep the original token (no change)

            # Record the original token's vocab id as the prediction target
            try:
                token_labels.append(tokenizer.vocab[token])
            except KeyError:
                token_labels.append(tokenizer.vocab["[UNK]"])
        else:
            # Not selected for masking — ignored by the loss function
            token_labels.append(-1)

    return masked_tokens, token_labels


# ---------------------------------------------------------------------------
# Masked Frame Modeling (MFM)
# ---------------------------------------------------------------------------


def mask_video_frames(video, max_video_length, mfm_probability=0.15):
    """Apply Masked Frame Modeling masking to pre-extracted video features.

    Analogous to MLM for text: randomly selected frames are zeroed out and
    the model must predict them from context.

    Args:
        video: Video feature array of shape
            ``(n_clips, max_frames, feature_dim)``.
        max_video_length: List of actual (unpadded) frame counts per clip,
            length ``n_clips``.
        mfm_probability: Probability of masking each real frame.

    Returns:
        masked_video: Copy of *video* with masked frames set to zero.
        video_labels_index: ``int64`` array of shape
            ``(n_clips, max_frames)``.  Contains the frame index for masked
            positions and ``-1`` for unmasked / padding positions.
    """
    n_clips = video.shape[0]
    feature_dim = video.shape[-1]

    video_labels_index = [[] for _ in range(n_clips)]
    masked_video = video.copy()

    for i in range(n_clips):
        for j in range(video.shape[1]):
            if j < max_video_length[i]:
                if random.random() < mfm_probability:
                    masked_video[i][j] = [0.0] * feature_dim
                    video_labels_index[i].append(j)
                else:
                    video_labels_index[i].append(-1)
            else:
                # Padding frames — never masked
                video_labels_index[i].append(-1)

    video_labels_index = np.array(video_labels_index, dtype=np.int64)
    return masked_video, video_labels_index


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_and_convert_tokens(token_ids, max_length, pad_value=0):
    """Pad a list of token ids to *max_length* with *pad_value*.

    Operates **in-place** on *token_ids* and returns a corresponding
    attention mask (1 for real tokens, 0 for padding).

    Args:
        token_ids: Mutable list of integer token ids.
        max_length: Desired length after padding.
        pad_value: Value used for padding slots.

    Returns:
        attention_mask: List of ``1`` / ``0`` of length *max_length*.
    """
    attention_mask = [1] * len(token_ids)
    while len(token_ids) < max_length:
        token_ids.append(pad_value)
        attention_mask.append(0)
    assert len(token_ids) == max_length
    return attention_mask
