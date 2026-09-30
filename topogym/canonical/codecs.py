"""Image codecs for the canonical dataset format.

PNG encoding and decoding need nothing beyond numpy and the standard
library (8-bit RGB and grayscale, 16-bit grayscale). JPEG needs Pillow.

Two array conventions live here so that every producer stores them
the same way (see :mod:`topogym.canonical.spec`):

- **Depth**: z-depth in metres, stored as a 16-bit grayscale PNG in
  :data:`~topogym.canonical.spec.DEPTH_UNIT_M` (2 mm) units, capped at
  :data:`~topogym.canonical.spec.DEPTH_MAX_CODE` (32767, i.e. 65.534 m);
  code 0 means no valid depth. The cap exists because LeRobot decodes
  16-bit PNGs as int16, so codes above 32767 would wrap negative.
- **Segmentation**: instance ids (0 = none) stored as a 16-bit
  grayscale PNG, ids at most 32767 for the same reason. What an id
  means is a per-episode table ``{id: {"category", "label"}}`` in the
  episode record, not in the image.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from topogym.canonical import spec

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def encode_png(img: np.ndarray) -> bytes:
    """PNG bytes for an image: (H, W, 3) uint8 RGB, (H, W) or (H, W, 1)
    uint8 grayscale, or (H, W) / (H, W, 1) uint16 grayscale."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim == 3 and img.shape[-1] == 3 and img.dtype == np.uint8:
        color, depth = 2, 8
    elif img.ndim == 2 and img.dtype == np.uint8:
        color, depth = 0, 8
    elif img.ndim == 2 and img.dtype == np.uint16:
        color, depth = 0, 16
    else:
        raise ValueError(f"cannot encode a {img.dtype} image of shape "
                         f"{img.shape} as PNG")
    h, w = img.shape[:2]
    pixels = np.ascontiguousarray(img)
    if depth == 16:
        pixels = pixels.astype(">u2")  # PNG is big-endian
    row = pixels.reshape(h, -1).view(np.uint8)
    raw = np.empty((h, 1 + row.shape[1]), dtype=np.uint8)
    raw[:, 0] = 0  # filter type: none
    raw[:, 1:] = row
    return (_SIGNATURE
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color,
                                          0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + _chunk(b"IEND", b""))


def _unfilter(data: np.ndarray, h: int, stride: int, bpp: int) -> np.ndarray:
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    pos = 0
    for y in range(h):
        kind = data[pos]
        line = data[pos + 1:pos + 1 + stride].astype(np.int32)
        pos += 1 + stride
        if kind == 0:
            cur = line
        elif kind == 2:
            cur = (line + prev) & 0xFF
        else:
            cur = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if kind == 1:
                    pred = a
                elif kind == 3:
                    pred = (a + b) >> 1
                elif kind == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else \
                        (b if pb <= pc else c)
                else:
                    raise ValueError(f"bad PNG filter type {kind}")
                cur[x] = (line[x] + pred) & 0xFF
        out[y] = cur
        prev = cur
    return out


def decode_png(data: bytes) -> np.ndarray:
    """Decode a non-interlaced 8-bit RGB/RGBA/grayscale or 16-bit
    grayscale PNG: (H, W, 3) or (H, W, 4) uint8, (H, W) uint8 or
    (H, W) uint16. Other PNGs need Pillow (:func:`decode_image`)."""
    if data[:8] != _SIGNATURE:
        raise ValueError("not a PNG")
    pos, idat, header = 8, [], None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError("PNG without IHDR")
    w, h, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 6: 4}.get(color)
    if channels is None or interlace or (depth == 16 and color != 0) \
            or depth not in (8, 16):
        raise ValueError("unsupported PNG variant; decode it with Pillow")
    bpp = channels * depth // 8
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    rows = _unfilter(raw, h, w * bpp, bpp)
    if depth == 16:
        return rows.view(">u2").reshape(h, w).astype(np.uint16)
    return rows.reshape(h, w, channels)[..., 0] if channels == 1 \
        else rows.reshape(h, w, channels)


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "JPEG needs Pillow: pip install 'topogym[jpeg]'") from exc
    return Image


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    """JPEG bytes for (H, W, 3) or (H, W) uint8 (needs Pillow)."""
    import io

    image = _pil()
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    buf = io.BytesIO()
    image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    """Decode PNG (without dependencies) or JPEG and other formats
    (with Pillow)."""
    if data[:8] == _SIGNATURE:
        try:
            return decode_png(data)
        except ValueError:
            pass
    import io

    image = _pil()
    return np.asarray(image.open(io.BytesIO(data)))


# -- depth -----------------------------------------------------------------------


def encode_depth(depth_m) -> np.ndarray:
    """Metres (any shape; NaN, inf or <= 0 for no return) to uint16
    codes in :data:`spec.DEPTH_UNIT_M` units, clipped to
    ``[1, DEPTH_MAX_CODE]`` for valid depth and 0 for invalid."""
    d = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    codes = np.rint(np.where(valid, d, 0.0) / spec.DEPTH_UNIT_M)
    codes = np.clip(codes, 1, spec.DEPTH_MAX_CODE)
    return np.where(valid, codes, 0).astype(np.uint16)


def decode_depth(codes, invalid: float = float("nan")) -> np.ndarray:
    """uint16 (or int16, as LeRobot decodes them) codes to float32
    metres; ``invalid`` (NaN by default) where there is no valid depth.
    Pass the value your live observations use (e.g. 0.0) to read back
    exactly what was observed."""
    c = np.asarray(codes).astype(np.int64)
    c = np.where(c < 0, c + 65536, c)  # undo an int16 wrap, if any
    out = c.astype(np.float32) * np.float32(spec.DEPTH_UNIT_M)
    return np.where(c > 0, out, np.float32(invalid)).astype(np.float32)


def encode_depth_png(depth_m) -> bytes:
    d = np.asarray(depth_m)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    return encode_png(encode_depth(d))


def decode_depth_png(data: bytes, invalid: float = float("nan")
                     ) -> np.ndarray:
    return decode_depth(decode_png(data), invalid)


# -- segmentation ---------------------------------------------------------------


def encode_segmentation(ids) -> np.ndarray:
    """Instance ids (0 = none) to uint16, refusing ids that would not
    survive an int16 decode."""
    a = np.asarray(ids)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.size and (a.min() < 0 or a.max() > spec.SEGMENTATION_MAX_ID):
        raise ValueError(
            f"segmentation ids must lie in [0, {spec.SEGMENTATION_MAX_ID}]; "
            f"got [{a.min()}, {a.max()}]")
    return a.astype(np.uint16)


def decode_segmentation(codes) -> np.ndarray:
    c = np.asarray(codes).astype(np.int64)
    return np.where(c < 0, c + 65536, c).astype(np.int32)


# -- other integer arrays ---------------------------------------------------------


def encode_array16(values) -> np.ndarray:
    """Integer codes in [0, ARRAY16_MAX] to uint16 (the generic 2-D
    array storage; the meaning of a code is the feature's to declare)."""
    a = np.asarray(values)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.size and (a.min() < 0 or a.max() > spec.ARRAY16_MAX):
        raise ValueError(f"array_png16 codes must lie in [0, "
                         f"{spec.ARRAY16_MAX}]; got [{a.min()}, {a.max()}]")
    return a.astype(np.uint16)


def decode_array16(codes) -> np.ndarray:
    c = np.asarray(codes).astype(np.int64)
    return np.where(c < 0, c + 65536, c).astype(np.int32)
