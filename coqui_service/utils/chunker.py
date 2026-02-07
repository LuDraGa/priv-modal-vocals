"""Text chunking for TTS synthesis.

Retrofitted from story_reels tts_v2/pipeline/chunker.py for Modal API use.
Simplified to focus on sentence-boundary chunking for Coqui XTTS.
"""

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using common punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(
    text: str,
    max_chars: int = 200,
    max_words: int = 60,
    min_chars: int = 40,
    preserve_sentence_boundaries: bool = True,
) -> List[str]:
    """Chunk text for TTS synthesis.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk (default 200 for XTTS safety)
        max_words: Maximum words per chunk
        min_chars: Minimum characters per chunk (merge small final chunks)
        preserve_sentence_boundaries: Split on sentence boundaries

    Returns:
        List of text chunks

    Note:
        XTTS v2 warns at 250 chars, so we default to 200 for safety.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Split into sentences if requested
    sentences = split_sentences(cleaned) if preserve_sentence_boundaries else [cleaned]

    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    current_words = 0

    def flush_current() -> None:
        """Flush current buffer to chunks."""
        nonlocal current, current_chars, current_words
        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)
        current = []
        current_chars = 0
        current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_chars = len(sentence)
        sentence_word_count = len(sentence_words)

        # If single sentence exceeds limits, split by words
        if sentence_word_count > max_words:
            for word in sentence_words:
                word_len = len(word)
                if (current_words + 1 > max_words) or (current_chars + word_len + 1 > max_chars):
                    flush_current()
                current.append(word)
                current_words += 1
                current_chars += word_len + 1
            continue

        # Check if adding sentence would exceed limits
        if (current_words + sentence_word_count > max_words) or \
           (current_chars + sentence_chars + 1 > max_chars):
            flush_current()

        current.append(sentence)
        current_words += sentence_word_count
        current_chars += sentence_chars + 1

    flush_current()

    # Merge tiny final chunk into previous chunk
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks
