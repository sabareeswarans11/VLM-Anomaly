"""Image resize, crop, and base64-encode helpers.

All cloud VLM APIs accept images as base64-encoded strings embedded in
the request body.  This module normalises images to a consistent size and
encoding before handing them to any backend.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

# Maximum dimension sent to cloud APIs.  Larger images cost more tokens
# and do not meaningfully improve detection accuracy.
MAX_SIDE_PX: int = 1024


def load_and_resize(image_path: Path, max_side: int = MAX_SIDE_PX) -> Image.Image:
    """Open an image and downscale it so its longest side ≤ ``max_side``.

    Does NOT upscale images that are already smaller.  Aspect ratio is
    preserved.  The result is always RGB (alpha channel stripped).

    Args:
        image_path: Path to any Pillow-readable image (PNG, JPG, …).
        max_side: Maximum pixel dimension on the longest side.

    Returns:
        A Pillow :class:`~PIL.Image.Image` in RGB mode.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    """Encode a Pillow image to JPEG bytes.

    Args:
        img: RGB Pillow image.
        quality: JPEG quality (1–95).

    Returns:
        Raw JPEG bytes.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def image_to_base64(image_path: Path, max_side: int = MAX_SIDE_PX) -> str:
    """Load, resize, and base64-encode an image as a plain string.

    Args:
        image_path: Path to the source image.
        max_side: Maximum pixel dimension before encoding.

    Returns:
        Base64-encoded JPEG string (no ``data:`` prefix).
    """
    img = load_and_resize(image_path, max_side)
    raw = to_jpeg_bytes(img)
    return base64.b64encode(raw).decode("ascii")


def image_to_data_url(image_path: Path, max_side: int = MAX_SIDE_PX) -> str:
    """Return a ``data:image/jpeg;base64,…`` URL for the image.

    Used by OpenAI-compatible APIs (Together.ai, Groq) that expect an
    ``image_url`` field with a data URI.

    Args:
        image_path: Path to the source image.
        max_side: Maximum pixel dimension before encoding.

    Returns:
        Full data URI string.
    """
    b64 = image_to_base64(image_path, max_side)
    return f"data:image/jpeg;base64,{b64}"
