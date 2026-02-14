#!/usr/bin/env python3
"""Generate seed corpus for the trace binary fuzzer.

Creates valid and near-valid .trc binary seeds so libFuzzer can mutate from
known-good structures instead of starting from random bytes.
"""
import json
import os
import struct
import uuid

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus", "fuzz_trace_binary")
os.makedirs(CORPUS_DIR, exist_ok=True)

def write_seed(name: str, data: bytes):
    path = os.path.join(CORPUS_DIR, name)
    with open(path, "wb") as f:
        f.write(data)
    print(f"  {name}: {len(data)} bytes")


# ── Valid minimal trace (1 entry) ──────────────────────────────────

def make_valid_trace(entries_json: list[dict], session_uuid: bytes = None) -> bytes:
    buf = bytearray()
    buf += b"TRCV"                                       # magic
    buf += struct.pack("<I", 1)                          # version
    if session_uuid is None:
        session_uuid = uuid.uuid4().bytes
    buf += session_uuid                                  # 16-byte UUID
    buf += struct.pack("<Q", len(entries_json))           # entry count
    for entry in entries_json:
        payload = json.dumps(entry, separators=(",", ":")).encode("utf-8")
        buf += struct.pack("<I", len(payload))
        buf += payload
    return bytes(buf)


entry_svd = {
    "seq": 0,
    "op": "svd_truncated",
    "timestamp_ns": 1700000000000000000,
    "duration_ns": 1000000,
    "input_hashes": {"A": "deadbeef"},
    "output_hashes": {"U": "cafebabe"},
    "params": {"chi_max": 20},
    "metrics": {"truncation_error": 1e-6},
}

entry_mpo = {
    "seq": 1,
    "op": "mpo_apply",
    "timestamp_ns": 1700000001000000000,
    "duration_ns": 500000,
    "input_hashes": {},
    "output_hashes": {},
    "params": {},
    "metrics": {},
}

print("Generating binary seeds:")
write_seed("valid_1entry.trc", make_valid_trace([entry_svd]))
write_seed("valid_2entry.trc", make_valid_trace([entry_svd, entry_mpo]))
write_seed("valid_empty.trc", make_valid_trace([]))

# Minimal entry (only required fields).
minimal_entry = {"seq": 0, "op": "custom", "timestamp_ns": 0}
write_seed("valid_minimal.trc", make_valid_trace([minimal_entry]))

# ── Edge-case seeds ────────────────────────────────────────────────

# Bad magic
write_seed("bad_magic.bin", b"XXXX" + b"\x01\x00\x00\x00" + b"\x00" * 24)

# Truncated after magic
write_seed("trunc_magic.bin", b"TRCV")

# Truncated after version
write_seed("trunc_version.bin", b"TRCV\x01\x00\x00\x00")

# Truncated mid-UUID
write_seed("trunc_uuid.bin", b"TRCV\x01\x00\x00\x00" + b"\xAB" * 8)

# Entry count = 1 but no entry data
header = b"TRCV\x01\x00\x00\x00" + b"\x00" * 16 + struct.pack("<Q", 1)
write_seed("missing_entry.bin", header)

# Entry with json_len = 0
header_zero = header + struct.pack("<I", 0)
write_seed("zero_json_len.bin", header_zero)

# Entry with enormous json_len (triggers OOM guard if any)
header_huge = header + struct.pack("<I", 0xFFFFFFFF)
write_seed("huge_json_len.bin", header_huge)

# Valid header + invalid JSON payload
header_bad_json = header + struct.pack("<I", 5) + b"{bad}"
write_seed("bad_json_payload.bin", header_bad_json)

# Wrong version
write_seed("wrong_version.bin", b"TRCV\x02\x00\x00\x00" + b"\x00" * 24)

# Empty file
write_seed("empty.bin", b"")

# Single byte
write_seed("one_byte.bin", b"\x00")

print(f"\nGenerated {len(os.listdir(CORPUS_DIR))} seed files in {CORPUS_DIR}")
