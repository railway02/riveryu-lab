#!/usr/bin/env python3
"""Companion exercises for the article: no third-party packages required.

Run: python computer_systems_first_principles_labs.py
All quantities follow explicitly stated toy-model assumptions; this is NOT a performance benchmark.
"""
from __future__ import annotations

from bisect import bisect_left
from json import loads
from math import exp, floor, isclose, isfinite, log2, sqrt
from struct import pack, unpack
from typing import Sequence


def round_half_away_from_zero(x: float) -> int:
    """For finite small values, emulate Rust f32::round tie direction in a toy model."""
    if not isfinite(x):
        raise ValueError("non-finite quantization input")
    return (1 if x >= 0 else -1) * floor(abs(x) + 0.5)


def quantize_groupwise_int4(values: Sequence[float], group_size: int):
    """Signed symmetric [-7,7]; one float scale per group; pack global nibble order."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if any(not isfinite(v) for v in values):
        raise ValueError("non-finite input")
    codes: list[int] = []
    scales: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        m = max(abs(v) for v in group)
        scale = m / 7 if m else 1.0
        scales.append(scale)
        codes.extend(max(-7, min(7, round_half_away_from_zero(v / scale))) for v in group)
    payload = bytearray((len(codes) + 1) // 2)
    for i, code in enumerate(codes):
        payload[i // 2] |= (code & 0xF) << (4 * (i % 2))
    return bytes(payload), scales, codes


def dequantize_groupwise_int4(payload: bytes, scales: Sequence[float],
                               count: int, group_size: int) -> list[float]:
    if len(payload) != (count + 1) // 2:
        raise ValueError("payload size mismatch")
    if len(scales) != (count + group_size - 1) // group_size:
        raise ValueError("scale size mismatch")
    out = []
    for i in range(count):
        nibble = (payload[i // 2] >> (4 * (i % 2))) & 0xF
        code = nibble - 16 if nibble >= 8 else nibble
        out.append(code * scales[i // group_size])
    return out


def lower_bound(a: Sequence[int], target: int) -> int:
    left, right = 0, len(a)
    while left < right:
        mid = left + (right - left) // 2
        if a[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


def translate_virtual_address(virtual_address: int, page_size: int,
                              page_table: dict[int, int]) -> int:
    vp, offset = divmod(virtual_address, page_size)
    if vp not in page_table:
        raise MemoryError("unmapped virtual page")
    return page_table[vp] * page_size + offset


def ipv4_fragment_payload(payload_bytes: int, mtu: int, ip_header: int = 20):
    """Toy IPv4 fragmentation with no options and a previously unfragmented datagram."""
    if payload_bytes < 0 or ip_header < 20 or mtu <= ip_header:
        raise ValueError("invalid lengths")
    max_payload = ((mtu - ip_header) // 8) * 8
    if max_payload <= 0 and payload_bytes > mtu - ip_header:
        raise ValueError("MTU too small for fragmentation")
    fragments = []
    offset_bytes = 0
    while offset_bytes < payload_bytes:
        remaining = payload_bytes - offset_bytes
        take = min(max_payload, remaining) if remaining > mtu - ip_header else remaining
        more = (offset_bytes + take) < payload_bytes
        fragments.append((take, take + ip_header, offset_bytes // 8, int(more)))
        offset_bytes += take
    return fragments


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def softmax_stable(scores: Sequence[float]) -> list[float]:
    m = max(scores)
    exps = [exp(v - m) for v in scores]
    z = sum(exps)
    return [a / z for a in exps]


def causal_attention_full(xs: Sequence[Sequence[float]]) -> list[list[float]]:
    # Identity Q, K, V projections; compute a separate causal softmax for each position.
    d = len(xs[0]); n = len(xs); out = []
    # Materialize ALL query-key scores as a dense matrix, then apply causal rows.
    # This purposely differs from the cached online implementation.
    scores_matrix = [[dot(xs[i], xs[j]) / sqrt(d) for j in range(n)] for i in range(n)]
    for i in range(n):
        # Mathematically equivalent to causal masking j > i with -infinity.
        p = softmax_stable(scores_matrix[i][:i+1])
        out.append([sum(p[j] * xs[j][k] for j in range(i + 1)) for k in range(d)])
    return out


def causal_attention_with_cache(xs: Sequence[Sequence[float]]) -> list[list[float]]:
    # Identity Q/K/V: cache only valid previous K and V; each step appends one.
    d = len(xs[0]); keys = []; values = []; out = []
    for x in xs:
        keys.append(list(x)); values.append(list(x))
        scores = [dot(x, key) / sqrt(d) for key in keys]
        p = softmax_stable(scores)
        out.append([sum(p[j] * values[j][k] for j in range(len(p))) for k in range(d)])
    return out



def gemv_row_major(weights: Sequence[float], rows: int, cols: int,
                   x: Sequence[float], out: list[float]) -> None:
    """Synchronous row-major reference; not a high-performance kernel."""
    if rows < 0 or cols < 0 or len(weights) != rows * cols or len(x) != cols or len(out) != rows:
        raise ValueError("shape mismatch")
    for i in range(rows):
        out[i] = sum(weights[i * cols + j] * x[j] for j in range(cols))


def cumulative_tcp_ack(chunks: Sequence[tuple[int, bytes]], initial_seq: int) -> list[int]:
    """Toy receiver: non-overlapping segments, no wrap-around and no FIN/SYN."""
    expected = initial_seq
    pending: dict[int, bytes] = {}
    acks = []
    for seq, body in chunks:
        if seq >= expected:
            pending[seq] = body
        while expected in pending:
            body = pending.pop(expected)
            expected += len(body)
        acks.append(expected)
    return acks


def test_byte_protocol_and_endianness() -> None:
    # The bit pattern for -6.5 is fixed; its byte order depends on representation convention.
    assert pack('>f', -6.5) == bytes.fromhex('c0d00000')
    assert pack('<f', -6.5) == bytes.fromhex('0000d0c0')
    assert unpack('<f', bytes.fromhex('0000d0c0')) == (-6.5,)

    body = '{"prompt":"请解释什么是 INT4 量化"}'
    encoded = body.encode('utf-8')
    assert len(body) == 27 and len(encoded) == 43
    # TCP delivers a byte stream. Segmentation is not a message boundary.
    chunks = [encoded[:3], encoded[3:9], encoded[9:27], encoded[27:]]
    joined = b''.join(chunks)
    assert len(joined) == 43
    assert loads(joined.decode('utf-8')) == {'prompt': '请解释什么是 INT4 量化'}

    codes = ['0', '10', '110', '111']
    assert all(not b.startswith(a) for i, a in enumerate(codes)
               for j, b in enumerate(codes) if i != j)
    probabilities = [1/2, 1/4, 1/8, 1/8]
    avg_len = sum(p * len(code) for p, code in zip(probabilities, codes))
    entropy = -sum(p * log2(p) for p in probabilities)
    assert avg_len == entropy == 1.75
    print('PASS little/big endian; UTF-8 43 bytes; message framing; entropy 1.75 bit/symbol')


def test_gemv_and_tcp_state() -> None:
    out = [0.0, 0.0]
    gemv_row_major([1, 2, 3, 4, 5, 6], 2, 3, [1, 0, 1], out)
    assert out == [4.0, 10.0]
    try:
        gemv_row_major([1, 2, 3], 2, 3, [1, 0, 1], out)
        raise AssertionError('shape error did not raise')
    except ValueError:
        pass
    chunks = [(1001, b'a' * 1000), (3001, b'c' * 1000), (2001, b'b' * 1000)]
    assert cumulative_tcp_ack(chunks, 1001) == [2001, 2001, 4001]
    print('PASS row-major GEMV and negative shape test; TCP cumulative ACK 2001/2001/4001')

def test_representation_and_quantization() -> None:
    assert pack('!f', -6.5).hex() == 'c0d00000'
    assert round_half_away_from_zero(2.5) == 3
    assert round_half_away_from_zero(-2.5) == -3
    x = [-7.0, -3.0, 7.0, -14.0, 6.0]
    payload, scales, codes = quantize_groupwise_int4(x, 3)
    assert codes == [-7, -3, 7, -7, 3]
    assert list(payload) == [0xD9, 0x97, 0x03]
    assert scales == [1.0, 2.0]
    assert dequantize_groupwise_int4(payload, scales, len(x), 3) == x
    print('PASS FP32 -6.5: 0xC0D00000; INT4 pack: D9 97 03; scales: 1, 2')


def test_data_structure_memory_and_ip() -> None:
    a = [1, 2, 2, 2, 5]
    for target in [-1, 1, 2, 3, 5, 6]:
        assert lower_bound(a, target) == bisect_left(a, target)
    assert translate_virtual_address(0x3123, 4096, {3: 9}) == 0x9123
    frags = ipv4_fragment_payload(2080, 1000, 20)
    assert frags == [(976, 996, 0, 1), (976, 996, 122, 1), (128, 148, 244, 0)]
    print('PASS lower_bound; virtual address 0x3123 -> 0x9123; IPv4 fragments 976/976/128')


def test_attention_and_future_mask() -> None:
    xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]]
    full = causal_attention_full(xs)
    cached = causal_attention_with_cache(xs)
    for row_a, row_b in zip(full, cached):
        for a, b in zip(row_a, row_b):
            assert isclose(a, b, rel_tol=1e-13, abs_tol=1e-13)
    changed = xs[:-1] + [[100.0, -100.0]]
    changed_output = causal_attention_full(changed)
    # The future position cannot influence the earlier causal outputs.
    for row_a, row_b in zip(full[:-1], changed_output[:-1]):
        for a, b in zip(row_a, row_b):
            assert isclose(a, b, rel_tol=1e-13, abs_tol=1e-13)
    print('PASS causal attention reference == append-only KV cache; future-token isolation')


def test_workload_math() -> None:
    parameters = 4096 * 4096
    fp32 = parameters * 4
    int4_group128 = parameters // 2 + parameters // 128 * 4
    assert fp32 / 1024**2 == 64
    assert int4_group128 / 1024**2 == 8.5
    kv = 2 * 32 * 1 * 8192 * 8 * 128 * 2
    assert kv == 2**30
    assert isclose(1 / (.9 + .1 / 10), 1.098901098901099)
    print('PASS FP32 64 MiB, INT4+scales 8.5 MiB; KV cache 1 GiB; Amdahl 1.099x')




def test_transfer_practice_v3() -> None:
    """Independent exercise numbers in the revised article: answer-key smoke tests."""
    assert pack('>f', -13.25).hex() == 'c1540000'
    payload, scales, codes = quantize_groupwise_int4([-7, -2, 7, -14, 6], 3)
    assert payload == bytes.fromhex('e9 97 03')
    assert scales == [1.0, 2.0] and codes == [-7, -2, 7, -7, 3]
    assert lower_bound([-2, 0, 0, 3, 7], 1) == 3
    assert lower_bound([-2, 0, 0, 3, 7], 8) == 5
    assert translate_virtual_address(0x5123, 8192, {2: 7}) == 0xF123
    assert ipv4_fragment_payload(2500 - 20, 900, 20) == [
        (880, 900, 0, 1), (880, 900, 110, 1), (720, 740, 220, 0)]
    assert 1250 * 8 / (20 * 10**6) == 0.0005
    window = (2*0.012 + 0.0005) / 0.0005
    assert isclose(window, 49, abs_tol=1e-12)
    assert (2**6 - 1) >= 49 and (2**(7 - 1)) >= 49
    kv = 2 * 24 * 2 * 4096 * 8 * 128 * 2
    assert kv == 768 * 2**20
    assert isclose(1 / (0.8 + 0.2/4), 1.1764705882352942)
    print('PASS changed exercise numbers: FP32, INT4, search, paging, IPv4, window, KV, Amdahl')


def test_prefill_decode_roofline_v3() -> None:
    """Idealized GEMM IO lower-bound arithmetic; not a device benchmark."""
    n = 4096
    FLOPS_S = 5e12
    BANDWIDTH_BYTES_S = 100e9
    ridge = FLOPS_S / BANDWIDTH_BYTES_S
    assert ridge == 50
    assert 4*n*n == 64*2**20
    for t, expected in [(1, 0.499756), (16, 7.937984),
                        (128, 60.235294), (512, 204.8)]:
        flops = 2*n*n*t
        moved = 4*n*n + 8*n*t  # read W and X once; write Y once
        assert isclose(flops / moved, expected, abs_tol=1e-5)
    t = 512
    flops = 2*n*n*t
    moved = 4*n*n + 8*n*t
    assert moved == 80*2**20
    assert isclose(flops / FLOPS_S * 1e3, 3.4359738368, abs_tol=1e-8)
    assert isclose(moved / BANDWIDTH_BYTES_S * 1e3, 0.8388608, abs_tol=1e-8)
    print('PASS GEMV/GEMM intensity: 0.50, 7.94, 60.24, 204.8 FLOP/B; unit conversions')


def test_prefix_cache_and_sse_v3() -> None:
    """Block alignment and application framing demos, not an HTTP/SSE implementation."""
    a = list('ABCDEFGHIJ')
    b = list('ABCDEFGXKL')
    block_size = 4
    # Only fully matching prefixes are eligible in this simple toy matcher.
    whole = min(len(a), len(b)) // block_size
    common_blocks = 0
    for block in range(whole):
        lo = block*block_size
        hi = lo+block_size
        if a[lo:hi] != b[lo:hi]:
            break
        common_blocks += 1
    assert common_blocks == 1
    # A stream may be cut into arbitrary transport-level pieces.
    body = 'data: {"delta":"你"}\n\ndata: {"delta":"好"}\n\n'
    wire_pieces = [body.encode('utf-8')[:2], body.encode('utf-8')[2:19], body.encode('utf-8')[19:]]
    restored = b''.join(wire_pieces).decode('utf-8')
    events = [block for block in restored.split('\n\n') if block]
    assert len(events) == 2
    assert loads(events[0][len('data: '):]) == {'delta':'你'}
    assert loads(events[1][len('data: '):]) == {'delta':'好'}
    print('PASS full-block common prefix 1; SSE event boundaries != arbitrary transport fragments')

if __name__ == '__main__':
    test_representation_and_quantization()
    test_byte_protocol_and_endianness()
    test_gemv_and_tcp_state()
    test_data_structure_memory_and_ip()
    test_attention_and_future_mask()
    test_workload_math()
    test_transfer_practice_v3()
    test_prefill_decode_roofline_v3()
    test_prefix_cache_and_sse_v3()
    print('All toy-model checks passed. This is not a StarLight, GPU or production benchmark.')
