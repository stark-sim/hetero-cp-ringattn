"""
Minimal bincode encoder/decoder compatible with Rust bincode 1.3 (default options).

Rust bincode 1.3 defaults:
- Endian: LittleEndian
- IntEncoding: Fixint (fixed-width)
- Enum tag: u32 (4 bytes LE)
- usize/i64/u64: 8 bytes LE on 64-bit platforms
"""

import struct
from typing import List, Tuple, Optional, Union


def encode_u32(v: int) -> bytes:
    return struct.pack("<I", v)


def encode_u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def encode_i64(v: int) -> bytes:
    return struct.pack("<q", v)


def encode_vec_u8(v: bytes) -> bytes:
    return encode_u64(len(v)) + v


def encode_vec_i64(v: List[int]) -> bytes:
    data = encode_u64(len(v))
    for x in v:
        data += encode_i64(x)
    return data


def decode_u32(data: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def decode_u64(data: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def decode_i64(data: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from("<q", data, offset)[0], offset + 8


def decode_vec_u8(data: bytes, offset: int) -> Tuple[bytes, int]:
    length, offset = decode_u64(data, offset)
    length = int(length)
    return data[offset:offset + length], offset + length


def decode_vec_i64(data: bytes, offset: int) -> Tuple[List[int], int]:
    length, offset = decode_u64(data, offset)
    length = int(length)
    vals = []
    for _ in range(length):
        v, offset = decode_i64(data, offset)
        vals.append(v)
    return vals, offset


# WorkerCommand tags (must match Rust enum order)
CMD_PREFILL = 0
CMD_DECODE = 1
CMD_DECODE_BATCH = 2
CMD_SYNC_GLOBAL_SEQ_LEN = 3
CMD_RELEASE_REQUEST = 4
CMD_SHUTDOWN = 5

# WorkerResponse tags (must match Rust enum order)
RESP_PREFILL_DONE = 0
RESP_DECODE_DONE = 1
RESP_DECODE_BATCH_DONE = 2
RESP_ERROR = 3


def encode_command(cmd_kind: str, **kwargs) -> bytes:
    """Encode WorkerCommand to bincode bytes."""
    if cmd_kind == "Prefill":
        request_id = kwargs["request_id"]
        chunk = kwargs["chunk"]
        seq_offset = kwargs["seq_offset"]
        data = encode_u32(CMD_PREFILL)
        data += encode_u64(request_id)
        data += encode_vec_i64(chunk)
        data += encode_i64(seq_offset)
        return data
    elif cmd_kind == "Decode":
        request_id = kwargs["request_id"]
        token = kwargs["token"]
        data = encode_u32(CMD_DECODE)
        data += encode_u64(request_id)
        data += encode_i64(token)
        return data
    elif cmd_kind == "DecodeBatch":
        request_tokens = kwargs["request_tokens"]
        data = encode_u32(CMD_DECODE_BATCH)
        data += encode_u64(len(request_tokens))
        for rid, tok in request_tokens:
            data += encode_u64(rid)
            data += encode_i64(tok)
        return data
    elif cmd_kind == "SyncGlobalSeqLen":
        request_id = kwargs["request_id"]
        global_seq_len = kwargs["global_seq_len"]
        data = encode_u32(CMD_SYNC_GLOBAL_SEQ_LEN)
        data += encode_u64(request_id)
        data += encode_u64(global_seq_len)
        return data
    elif cmd_kind == "ReleaseRequest":
        request_id = kwargs["request_id"]
        data = encode_u32(CMD_RELEASE_REQUEST)
        data += encode_u64(request_id)
        return data
    elif cmd_kind == "Shutdown":
        return encode_u32(CMD_SHUTDOWN)
    else:
        raise ValueError(f"unknown command: {cmd_kind}")


def decode_command(data: bytes) -> dict:
    """Decode bincode bytes to WorkerCommand dict."""
    tag, offset = decode_u32(data, 0)
    if tag == CMD_PREFILL:
        request_id, offset = decode_u64(data, offset)
        chunk, offset = decode_vec_i64(data, offset)
        seq_offset, offset = decode_i64(data, offset)
        return {"kind": "Prefill", "request_id": int(request_id), "chunk": chunk, "seq_offset": seq_offset}
    elif tag == CMD_DECODE:
        request_id, offset = decode_u64(data, offset)
        token, offset = decode_i64(data, offset)
        return {"kind": "Decode", "request_id": int(request_id), "token": token}
    elif tag == CMD_DECODE_BATCH:
        length, offset = decode_u64(data, offset)
        request_tokens = []
        for _ in range(length):
            rid, offset = decode_u64(data, offset)
            tok, offset = decode_i64(data, offset)
            request_tokens.append((int(rid), tok))
        return {"kind": "DecodeBatch", "request_tokens": request_tokens}
    elif tag == CMD_SYNC_GLOBAL_SEQ_LEN:
        request_id, offset = decode_u64(data, offset)
        global_seq_len, offset = decode_u64(data, offset)
        return {"kind": "SyncGlobalSeqLen", "request_id": int(request_id), "global_seq_len": int(global_seq_len)}
    elif tag == CMD_RELEASE_REQUEST:
        request_id, offset = decode_u64(data, offset)
        return {"kind": "ReleaseRequest", "request_id": int(request_id)}
    elif tag == CMD_SHUTDOWN:
        return {"kind": "Shutdown"}
    else:
        raise ValueError(f"unknown command tag: {tag}")


def encode_response(resp_kind: str, **kwargs) -> bytes:
    """Encode WorkerResponse to bincode bytes."""
    if resp_kind == "PrefillDone":
        request_id = kwargs["request_id"]
        last_logits_bytes = kwargs["last_logits_bytes"]
        global_seq_len = kwargs["global_seq_len"]
        data = encode_u32(RESP_PREFILL_DONE)
        data += encode_u64(request_id)
        data += encode_vec_u8(last_logits_bytes)
        data += encode_u64(global_seq_len)
        return data
    elif resp_kind == "DecodeDone":
        request_id = kwargs["request_id"]
        logits_bytes = kwargs["logits_bytes"]
        data = encode_u32(RESP_DECODE_DONE)
        data += encode_u64(request_id)
        data += encode_vec_u8(logits_bytes)
        return data
    elif resp_kind == "DecodeBatchDone":
        request_logits = kwargs["request_logits"]
        data = encode_u32(RESP_DECODE_BATCH_DONE)
        data += encode_u64(len(request_logits))
        for rid, lbs in request_logits:
            data += encode_u64(rid)
            data += encode_vec_u8(lbs)
        return data
    elif resp_kind == "Error":
        request_id = kwargs["request_id"]
        msg = kwargs["msg"].encode()
        data = encode_u32(RESP_ERROR)
        data += encode_u64(request_id)
        data += encode_vec_u8(msg)
        return data
    else:
        raise ValueError(f"unknown response: {resp_kind}")


def decode_response(data: bytes) -> dict:
    """Decode bincode bytes to WorkerResponse dict."""
    tag, offset = decode_u32(data, 0)
    if tag == RESP_PREFILL_DONE:
        request_id, offset = decode_u64(data, offset)
        last_logits_bytes, offset = decode_vec_u8(data, offset)
        global_seq_len, offset = decode_u64(data, offset)
        return {
            "kind": "PrefillDone",
            "request_id": int(request_id),
            "last_logits_bytes": last_logits_bytes,
            "global_seq_len": int(global_seq_len),
        }
    elif tag == RESP_DECODE_DONE:
        request_id, offset = decode_u64(data, offset)
        logits_bytes, offset = decode_vec_u8(data, offset)
        return {"kind": "DecodeDone", "request_id": int(request_id), "logits_bytes": logits_bytes}
    elif tag == RESP_DECODE_BATCH_DONE:
        length, offset = decode_u64(data, offset)
        request_logits = []
        for _ in range(length):
            rid, offset = decode_u64(data, offset)
            lbs, offset = decode_vec_u8(data, offset)
            request_logits.append((int(rid), lbs))
        return {"kind": "DecodeBatchDone", "request_logits": request_logits}
    elif tag == RESP_ERROR:
        request_id, offset = decode_u64(data, offset)
        msg_bytes, offset = decode_vec_u8(data, offset)
        return {"kind": "Error", "request_id": int(request_id), "msg": msg_bytes.decode()}
    else:
        raise ValueError(f"unknown response tag: {tag}")


def encode_handshake(domain_id: int, capacity_mb: int) -> bytes:
    """Encode WorkerHandshake to 16 bytes."""
    return encode_u64(domain_id) + encode_u64(capacity_mb)


def decode_handshake(data: bytes) -> Tuple[int, int]:
    """Decode 16-byte WorkerHandshake."""
    domain_id, offset = decode_u64(data, 0)
    capacity_mb, offset = decode_u64(data, offset)
    return int(domain_id), int(capacity_mb)
