"""Allocator unit tests: coalescing, compaction and view-safe moves."""

from __future__ import annotations

from array import array

import pytest

from memdry.memory import Allocator, OutOfSpace


def fill(alloc: Allocator, storage, value: float) -> None:
    alloc.buffer[storage.offset : storage.offset + storage.length] = array(
        "f", [value] * storage.length
    )


def values_of(alloc: Allocator, storage) -> list[float]:
    return list(alloc.buffer[storage.offset : storage.offset + storage.length])


def test_first_fit_and_peak():
    alloc = Allocator(12)
    a, _ = alloc.allocate(3, "a", 0)
    b, _ = alloc.allocate(3, "b", 1)
    assert (a.offset, b.offset) == (0, 3)
    assert alloc.occupied == 6
    assert alloc.peak == 6
    c, _ = alloc.allocate(1, "c", 2)
    assert c.offset == 6
    assert alloc.peak == 7
    alloc.release(b)
    assert alloc.total_free() == 8
    assert alloc.largest_free() == 5
    d, compaction = alloc.allocate(4, "d", 3)
    # First-fit reuses the largest hole; no compaction needed.
    assert compaction is None
    assert d.offset == 7
    assert alloc.peak == 8  # a(3) + c(1) + d(4)


def test_release_coalesces_neighbours():
    alloc = Allocator(12)
    a, _ = alloc.allocate(4, "a", 0)
    b, _ = alloc.allocate(4, "b", 1)
    c, _ = alloc.allocate(4, "c", 2)
    alloc.release(a)
    alloc.release(c)
    assert sorted(block[1] for block in alloc.free_blocks) == [4, 4]
    alloc.release(b)
    assert alloc.free_blocks == [[0, 12]]
    assert alloc.occupied == 0


def test_fragmented_space_compacts_and_preserves_values():
    alloc = Allocator(16)
    a, _ = alloc.allocate(4, "a", 0)
    b, _ = alloc.allocate(4, "b", 1)
    c, _ = alloc.allocate(4, "c", 2)
    fill(alloc, a, 1.25)
    fill(alloc, b, 2.5)
    fill(alloc, c, 3.75)
    alloc.release(b)
    assert alloc.largest_free() == 4
    # 5 elements do not fit in any single 4-element hole, though 8 are free.
    d, compaction = alloc.allocate(5, "d", 3)
    assert compaction is not None
    assert compaction["moved"] == 1
    assert compaction["moved_elements"] == 4
    assert alloc.compactions == 1
    assert d.offset == 8
    assert values_of(alloc, a) == [1.25] * 4
    assert values_of(alloc, c) == [3.75] * 4


def test_compaction_leaves_already_packed_leading_block_in_place():
    alloc = Allocator(12)
    a, _ = alloc.allocate(4, "a", 0)
    b, _ = alloc.allocate(2, "b", 1)
    c, _ = alloc.allocate(4, "c", 2)
    fill(alloc, c, 5.0)
    alloc.release(b)
    assert alloc.largest_free() == 2
    d, compaction = alloc.allocate(3, "d", 3)
    assert compaction is not None
    # a was already at offset 0; only c had to move, from 6 to 4.
    assert a.offset == 0
    assert c.offset == 4
    assert compaction["moved"] == 1
    assert values_of(alloc, c) == [5.0] * 4
    assert d.offset == 8


def test_refused_when_total_free_is_too_small():
    alloc = Allocator(10)
    a, _ = alloc.allocate(4, "a", 0)
    b, _ = alloc.allocate(4, "b", 1)
    with pytest.raises(OutOfSpace) as info:
        alloc.allocate(4, "c", 2)
    exc = info.value
    assert exc.needed == 4
    assert exc.free_total == 2
    assert alloc.compactions == 0  # no pointless move when total is short
    assert len(alloc.live_storages()) == 2


def test_allocation_larger_than_capacity_is_refused():
    alloc = Allocator(4)
    with pytest.raises(OutOfSpace):
        alloc.allocate(5, "big", 0)
    assert alloc.free_blocks == [[0, 4]]


def test_reference_counting_keeps_shared_storage():
    alloc = Allocator(8)
    src, _ = alloc.allocate(4, "src", 0)
    fill(alloc, src, 7.0)
    src.refs += 1  # a slice view now shares the block
    assert alloc.release(src) is False
    assert alloc.live_storages() == [src]
    assert values_of(alloc, src) == [7.0] * 4
    assert alloc.release(src) is True
    assert alloc.live_storages() == []
    assert alloc.free_blocks == [[0, 8]]


def test_compaction_moves_shared_storage_once_keeping_contents():
    alloc = Allocator(12)
    src, _ = alloc.allocate(4, "src", 0)
    fill(alloc, src, 9.0)
    src.refs += 1  # a slice view shares the same block
    other, _ = alloc.allocate(2, "other", 1)
    tail, _ = alloc.allocate(4, "tail", 2)
    fill(alloc, tail, 2.0)
    alloc.release(other)
    assert alloc.largest_free() == 2  # [4,2] and [10,2] are both too small
    fresh, compaction = alloc.allocate(3, "fresh", 3)
    assert compaction is not None
    # The shared block at offset 0 stays; tail moves 6 -> 4 as one unit.
    assert src.offset == 0
    assert tail.offset == 4
    assert compaction["moved"] == 1
    assert src.refs == 2  # source handle plus the view
    assert values_of(alloc, src) == [9.0] * 4
    assert values_of(alloc, tail) == [2.0] * 4
    assert fresh.offset == 8


def test_layout_reports_views_and_free_gaps():
    alloc = Allocator(10)
    a, _ = alloc.allocate(4, "a", 0)
    alloc.release(a)
    b, _ = alloc.allocate(2, "b", 1)
    layout = alloc.layout()
    assert layout == [
        {"kind": "live", "start": 0, "length": 2, "storage": b.sid, "label": "b",
         "color": 1, "refs": 1, "views": []},
        {"kind": "free", "start": 2, "length": 8, "label": "空闲"},
    ]
