"""
protocol.py  --  Shared wire protocol for the Model-Extraction project (Group 9).

This is the SHARED file: the victim server (Pritu) and the attack client (Arian)
both import it so that every byte on the wire is packed/unpacked identically.
It implements exactly the framing described in the design report:

  * A fixed 12-byte header, big-endian, struct format ">HBBII".
  * A length-prefixed application-layer frame on top of TCP (which gives a
    reliable byte stream but no message boundaries), so we always know how
    many payload bytes follow and can reassemble across TCP segmentation.

Header (12 bytes)                     struct ">HBBII"
    off field        size type    meaning
    0   MAGIC        2 B  uint16  0x4D58 ("MX"): version/sanity guard
    2   VERSION      1 B  uint8   protocol version = 1
    3   MSG_TYPE     1 B  uint8   1=REQUEST, 2=RESPONSE, 3=ERROR
    4   REQUEST_ID   4 B  uint32  matches a response to its request
    8   PAYLOAD_LEN  4 B  uint32  number of payload bytes that follow

REQUEST payload (attacker -> victim)  struct ">HHB" + float32[]
    H   2 B uint16  image height
    W   2 B uint16  image width
    C   1 B uint8   channels (1 = grayscale)
    PIXELS  H*W*C*4 B  float32[] row-major, normalized to [0,1], big-endian

RESPONSE payload (victim -> attacker) struct ">HB" + float32[]
    K     2 B uint16  number of classes
    FLAGS 1 B uint8   bit0=probs present, bit1=top-k truncated, bit2=noised
    PROBS K*4 B float32[]  class probabilities after the defence layer

ERROR payload                          struct ">H"
    CODE  2 B uint16  error code (see ERR_* below)

All multi-byte integer fields AND the float arrays are big-endian (network
order). numpy dtype ">f4" is used for the pixel/prob arrays so the float bytes
are also in network order.
"""

import socket
import struct

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MAGIC = 0x4D58          # "MX"
VERSION = 1

# Message types
MSG_REQUEST = 1
MSG_RESPONSE = 2
MSG_ERROR = 3

# RESPONSE flag bits
FLAG_PROBS = 0b001      # bit0: a probability vector is present
FLAG_TOPK = 0b010       # bit1: only top-k classes are non-zero (truncated)
FLAG_NOISED = 0b100     # bit2: calibrated noise was added by the defence

# ERROR codes
ERR_BAD_MAGIC = 1
ERR_BAD_VERSION = 2
ERR_BAD_TYPE = 3
ERR_RATE_LIMIT = 4      # per-client quota exceeded (report's defence 6.3)
ERR_MALFORMED = 5

# Struct formats
HEADER_FMT = ">HBBII"           # MAGIC, VERSION, MSG_TYPE, REQUEST_ID, PAYLOAD_LEN
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 12
REQ_HDR_FMT = ">HHB"            # H, W, C
REQ_HDR_SIZE = struct.calcsize(REQ_HDR_FMT)  # 5
RESP_HDR_FMT = ">HB"            # K, FLAGS
RESP_HDR_SIZE = struct.calcsize(RESP_HDR_FMT)  # 3
ERR_FMT = ">H"                  # CODE
ERR_SIZE = struct.calcsize(ERR_FMT)

# numpy big-endian float32 for pixel / probability arrays
BE_F32 = np.dtype(">f4")


class ProtocolError(Exception):
    """Raised when a frame cannot be parsed (bad magic/version/type/length)."""


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

def pack_header(msg_type: int, request_id: int, payload_len: int) -> bytes:
    """Build the 12-byte common header."""
    return struct.pack(HEADER_FMT, MAGIC, VERSION, msg_type, request_id, payload_len)


def unpack_header(raw: bytes):
    """
    Parse a 12-byte header. Returns (msg_type, request_id, payload_len).
    Raises ProtocolError on bad magic or version so the caller can drop it.
    """
    if len(raw) != HEADER_SIZE:
        raise ProtocolError(f"header must be {HEADER_SIZE} bytes, got {len(raw)}")
    magic, version, msg_type, request_id, payload_len = struct.unpack(HEADER_FMT, raw)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:04X} (expected 0x{MAGIC:04X})")
    if version != VERSION:
        raise ProtocolError(f"bad version {version} (expected {VERSION})")
    return msg_type, request_id, payload_len


# --------------------------------------------------------------------------- #
# Frame encoders (return the full frame: header + payload)
# --------------------------------------------------------------------------- #

def encode_request(image: np.ndarray, request_id: int) -> bytes:
    """
    Encode one image into a REQUEST frame.

    image: float array in [0,1], shape (H, W) for grayscale, (H, W, C), or
           (C, H, W). It is stored row-major as (H, W, C).
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:                      # (H, W) -> (H, W, 1)
        arr = arr[:, :, None]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    elif arr.ndim != 3:
        raise ValueError(f"image must be 2D or 3D, got shape {arr.shape}")

    h, w, c = arr.shape
    pixels = np.ascontiguousarray(arr).astype(BE_F32).tobytes()
    payload = struct.pack(REQ_HDR_FMT, h, w, c) + pixels
    return pack_header(MSG_REQUEST, request_id, len(payload)) + payload


def encode_response(probs: np.ndarray, request_id: int, flags: int = FLAG_PROBS) -> bytes:
    """Encode a probability vector into a RESPONSE frame."""
    p = np.asarray(probs, dtype=np.float32).ravel()
    k = p.shape[0]
    payload = struct.pack(RESP_HDR_FMT, k, flags) + p.astype(BE_F32).tobytes()
    return pack_header(MSG_RESPONSE, request_id, len(payload)) + payload


def encode_error(request_id: int, code: int) -> bytes:
    """Encode an ERROR frame carrying a numeric code."""
    payload = struct.pack(ERR_FMT, code)
    return pack_header(MSG_ERROR, request_id, len(payload)) + payload


# --------------------------------------------------------------------------- #
# Payload decoders
# --------------------------------------------------------------------------- #

def decode_request(payload: bytes) -> np.ndarray:
    """Decode a REQUEST payload into a float32 image of shape (H, W, C) in [0,1]."""
    if len(payload) < REQ_HDR_SIZE:
        raise ProtocolError("request payload shorter than its sub-header")
    h, w, c = struct.unpack(REQ_HDR_FMT, payload[:REQ_HDR_SIZE])
    expected = REQ_HDR_SIZE + h * w * c * 4
    if len(payload) != expected:
        raise ProtocolError(f"request payload len {len(payload)} != expected {expected}")
    pixels = np.frombuffer(payload[REQ_HDR_SIZE:], dtype=BE_F32).astype(np.float32)
    return pixels.reshape(h, w, c)


def decode_response(payload: bytes):
    """Decode a RESPONSE payload. Returns (probs float32 (K,), flags int)."""
    if len(payload) < RESP_HDR_SIZE:
        raise ProtocolError("response payload shorter than its sub-header")
    k, flags = struct.unpack(RESP_HDR_FMT, payload[:RESP_HDR_SIZE])
    expected = RESP_HDR_SIZE + k * 4
    if len(payload) != expected:
        raise ProtocolError(f"response payload len {len(payload)} != expected {expected}")
    probs = np.frombuffer(payload[RESP_HDR_SIZE:], dtype=BE_F32).astype(np.float32)
    return probs, flags


def decode_error(payload: bytes) -> int:
    """Decode an ERROR payload into its code."""
    if len(payload) < ERR_SIZE:
        raise ProtocolError("error payload too short")
    (code,) = struct.unpack(ERR_FMT, payload[:ERR_SIZE])
    return code


# --------------------------------------------------------------------------- #
# Socket helpers: length-prefixed reassembly off the TCP stream
# --------------------------------------------------------------------------- #

def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """
    Read exactly n bytes from the socket, looping over recv() because TCP may
    deliver the message split across several segments or coalesced with others.
    Raises ConnectionError if the peer closes early.
    """
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed connection mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket):
    """
    Read one whole frame off the stream.

    Returns (msg_type, request_id, payload_bytes). Reads exactly 12 header
    bytes, parses PAYLOAD_LEN, then reads exactly that many payload bytes.
    """
    header = recv_exactly(sock, HEADER_SIZE)
    msg_type, request_id, payload_len = unpack_header(header)
    payload = recv_exactly(sock, payload_len) if payload_len else b""
    return msg_type, request_id, payload


def send_frame(sock: socket.socket, frame: bytes) -> None:
    """Send a fully-encoded frame (header + payload)."""
    sock.sendall(frame)


# --------------------------------------------------------------------------- #
# Tiny self-test: pack -> unpack round trip (no sockets involved)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(1337)

    img = rng.random((28, 28), dtype=np.float32)
    req = encode_request(img, request_id=1)
    mt, rid, payload = MSG_REQUEST, 1, req[HEADER_SIZE:]
    # verify header parse
    hmt, hrid, hlen = unpack_header(req[:HEADER_SIZE])
    assert (hmt, hrid, hlen) == (MSG_REQUEST, 1, len(payload)), "header mismatch"
    back = decode_request(payload)
    assert np.allclose(back[:, :, 0], img, atol=1e-6), "image round-trip failed"
    assert len(req) == HEADER_SIZE + 5 + 28 * 28 * 4 == 12 + 3141, "frame size mismatch"

    probs = rng.random(10, dtype=np.float32)
    probs /= probs.sum()
    resp = encode_response(probs, request_id=1, flags=FLAG_PROBS)
    rmt, rrid, rpayload = unpack_header(resp[:HEADER_SIZE])[0], 1, resp[HEADER_SIZE:]
    p2, flags = decode_response(rpayload)
    assert np.allclose(p2, probs, atol=1e-6), "probs round-trip failed"
    assert flags == FLAG_PROBS

    err = encode_error(7, ERR_RATE_LIMIT)
    assert decode_error(err[HEADER_SIZE:]) == ERR_RATE_LIMIT

    print("protocol.py self-test OK  (header=12B, 28x28 frame =", len(req), "bytes)")
