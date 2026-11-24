"""Fixed-buffer allocator and executor for the memory dry run.

The whole run uses one ``array('f')`` of ``capacity`` elements.  Every tensor
(input, intermediate, output) lives in that buffer.  ``add`` and ``relu``
allocate a fresh contiguous span for their result; ``slice`` creates a view
that shares the source's whole storage block, so it costs no extra elements.

If a request does not fit but the *total* free space is large enough, the
allocator compacts live blocks towards the start of the buffer (moving the
bytes inside the buffer and keeping the relative offsets of views intact) and
retries.  Only when the total free space is too small does the run fail, and
the failure carries the exact spot and the number of elements required.
"""

from __future__ import annotations

import math
import struct
from array import array
from dataclasses import dataclass

from .model import Plan, fmt_value

PALETTE_SIZE = 8


class OutOfSpace(Exception):
    """A request cannot be satisfied even after compaction."""

    def __init__(
        self, needed: int, free_total: int, largest_free: int, compactions: int
    ):
        self.needed = needed
        self.free_total = free_total
        self.largest_free = largest_free
        self.compactions = compactions
        super().__init__(
            f"need {needed}, free {free_total}, largest {largest_free}"
        )


def _round_f32(value: float) -> float:
    """IEEE-754 binary32 round-to-nearest (overflow becomes +/-inf)."""
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


@dataclass
class Storage:
    sid: int
    length: int
    label: str
    color: int
    offset: int = -1
    refs: int = 0

    @property
    def live(self) -> bool:
        return self.refs > 0 and self.offset >= 0


@dataclass
class TensorHandle:
    name: str
    storage: Storage
    offset: int  # relative to storage.offset
    length: int
    role: str  # input / result / view
    color: int
    slice_of: str | None = None
    released: bool = False


class Allocator:
    """First-fit allocator over one fixed float32 buffer, with compaction."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = array("f", [0.0]) * capacity
        self.free_blocks: list[list[int]] = [[0, capacity]]
        self.occupied = 0
        self.peak = 0
        self.peak_at = "开始"
        self.storages: list[Storage] = []
        self.compactions = 0
        self.moved_elements = 0
        self.allocations = 0

    # -- inspection ---------------------------------------------------------
    def total_free(self) -> int:
        return sum(block[1] for block in self.free_blocks)

    def largest_free(self) -> int:
        return max((block[1] for block in self.free_blocks), default=0)

    def live_storages(self) -> list[Storage]:
        return sorted((s for s in self.storages if s.live), key=lambda s: s.offset)

    def layout(self, handles: dict[str, TensorHandle] | None = None) -> list[dict]:
        views: dict[int, list[dict]] = {}
        if handles:
            for handle in handles.values():
                if handle.role == "view" and not handle.released:
                    views.setdefault(handle.storage.sid, []).append(
                        {
                            "name": handle.name,
                            "rel_start": handle.offset,
                            "length": handle.length,
                            "slice_of": handle.slice_of,
                        }
                    )
        entries: list[dict] = []
        for storage in self.live_storages():
            entries.append(
                {
                    "kind": "live",
                    "start": storage.offset,
                    "length": storage.length,
                    "storage": storage.sid,
                    "label": storage.label,
                    "color": storage.color,
                    "refs": storage.refs,
                    "views": views.get(storage.sid, []),
                }
            )
        for start, length in self.free_blocks:
            if length:
                entries.append(
                    {"kind": "free", "start": start, "length": length, "label": "空闲"}
                )
        entries.sort(key=lambda entry: entry["start"])
        return entries

    # -- allocation ---------------------------------------------------------
    def allocate(
        self, length: int, label: str, color: int
    ) -> tuple[Storage, dict | None]:
        if length < 1:
            raise ValueError("一次分配至少需要 1 个元素")
        if length > self.capacity or self.total_free() < length:
            raise OutOfSpace(
                length, self.total_free(), self.largest_free(), self.compactions
            )
        compaction = None
        index = self._find_block(length)
        if index is None:
            # Fragmented but the total free space is enough: pack and retry.
            compaction = self.compact()
            index = self._find_block(length)
            if index is None:  # defensive: a full compaction leaves one block
                raise OutOfSpace(
                    length, self.total_free(), self.largest_free(), self.compactions
                )
        start, block_len = self.free_blocks[index]
        self.free_blocks[index] = [start + length, block_len - length]
        if self.free_blocks[index][1] == 0:
            del self.free_blocks[index]
        storage = Storage(
            sid=len(self.storages),
            length=length,
            label=label,
            color=color,
            offset=start,
            refs=1,
        )
        self.storages.append(storage)
        self.occupied += length
        self.allocations += 1
        if self.occupied > self.peak:
            self.peak = self.occupied
            self.peak_at = label
        return storage, compaction

    def _find_block(self, length: int) -> int | None:
        for index, (_, block_len) in enumerate(self.free_blocks):
            if block_len >= length:
                return index
        return None

    def compact(self) -> dict:
        """Move live blocks to the front, in place, preserving their contents."""
        live = self.live_storages()
        before_largest = self.largest_free()
        moved = 0
        moved_elements = 0
        cursor = 0
        for storage in live:
            if storage.offset != cursor:
                self.buffer[cursor : cursor + storage.length] = self.buffer[
                    storage.offset : storage.offset + storage.length
                ]
                storage.offset = cursor
                moved += 1
                moved_elements += storage.length
            cursor += storage.length
        self.free_blocks = (
            [[cursor, self.capacity - cursor]] if cursor < self.capacity else []
        )
        self.compactions += 1
        self.moved_elements += moved_elements
        return {
            "storage_count": len(live),
            "moved": moved,
            "moved_elements": moved_elements,
            "before_largest": before_largest,
            "after_largest": self.largest_free(),
            "after_free": self.total_free(),
        }

    def release(self, storage: Storage) -> bool:
        """Drop one reference; return True when the storage was freed."""
        if storage.refs <= 0:
            raise RuntimeError("重复释放存储块")
        storage.refs -= 1
        if storage.refs > 0:
            return False
        self.occupied -= storage.length
        insertion = len(self.free_blocks)
        for index, (block_start, _) in enumerate(self.free_blocks):
            if block_start > storage.offset:
                insertion = index
                break
        self.free_blocks.insert(insertion, [storage.offset, storage.length])
        self._coalesce(insertion)
        return True

    def _coalesce(self, index: int) -> None:
        blocks = self.free_blocks
        index = max(0, min(index, len(blocks) - 1))
        if (
            index + 1 < len(blocks)
            and blocks[index][0] + blocks[index][1] == blocks[index + 1][0]
        ):
            blocks[index][1] += blocks[index + 1][1]
            del blocks[index + 1]
        if index > 0 and blocks[index - 1][0] + blocks[index - 1][1] == blocks[index][0]:
            blocks[index - 1][1] += blocks[index][1]
            del blocks[index]
            index -= 1


class Executor:
    def __init__(self, plan: Plan):
        self.plan = plan
        self.alloc = Allocator(plan.capacity)
        self.handles: dict[str, TensorHandle] = {}
        self.steps: list[dict] = []
        self.warnings = list(plan.warnings)
        self.overflow_values = 0
        self.last_use: dict[str, int] = {}
        for index, op in enumerate(plan.ops):
            for name in op.operands:
                self.last_use[name] = index
        self.output_set = set(plan.outputs)
        self.color_seq = 0
        self.current_action: dict | None = None

    # -- helpers ------------------------------------------------------------
    def _new_color(self) -> int:
        color = self.color_seq % PALETTE_SIZE
        self.color_seq += 1
        return color

    def _buffered_values(self, handle: TensorHandle) -> list[float]:
        start = handle.storage.offset + handle.offset
        return list(self.alloc.buffer[start : start + handle.length])

    def _store(self, handle: TensorHandle, values: list[float]) -> None:
        start = handle.storage.offset + handle.offset
        self.alloc.buffer[start : start + handle.length] = array("f", values)

    def _step(
        self,
        title: str,
        detail: str,
        *,
        kind: str,
        op: dict | None = None,
        allocated: list[dict] | None = None,
        freed: list[dict] | None = None,
        view: dict | None = None,
        compaction: dict | None = None,
    ) -> None:
        self.steps.append(
            {
                "index": len(self.steps) + 1,
                "title": title,
                "detail": detail,
                "kind": kind,
                "op": op,
                "allocated": allocated or [],
                "freed": freed or [],
                "view": view,
                "compaction": compaction,
                "occupancy": self.alloc.occupied,
                "percent": round(100.0 * self.alloc.occupied / self.plan.capacity, 2),
                "free": self.alloc.total_free(),
                "largest_free": self.alloc.largest_free(),
            }
        )

    def _release(self, name: str) -> dict | None:
        handle = self.handles.get(name)
        if handle is None or handle.released:
            return None
        handle.released = True
        storage_freed = self.alloc.release(handle.storage)
        return {
            "name": name,
            "role": handle.role,
            "storage": handle.storage.sid,
            "storage_freed": storage_freed,
        }

    def _release_if_done(self, name: str, op_index: int) -> dict | None:
        """Release a value after op *op_index* when it is not an output and unused later."""
        if name in self.output_set:
            return None
        if self.last_use.get(name, -99) > op_index:
            return None
        return self._release(name)

    def _release_now_if_dead(self, name: str) -> dict | None:
        """Release a just-produced value that nobody consumes and that is not an output."""
        if name in self.output_set or name in self.last_use:
            return None
        return self._release(name)

    def _live_names(self) -> list[str]:
        return [handle.name for handle in self.handles.values() if not handle.released]

    # -- run ----------------------------------------------------------------
    def run(self) -> dict:
        try:
            self._run_inputs()
            self._run_ops()
        except OutOfSpace as exc:
            return self._finish(failure=self._failure(exc))
        return self._finish(failure=None)

    def _run_inputs(self) -> None:
        for spec in self.plan.inputs:
            action = {
                "name": spec.name,
                "kind": "input",
                "line": spec.line,
                "operands": [],
                "length": spec.length,
            }
            self.current_action = action
            color = self._new_color()
            storage, compaction = self.alloc.allocate(
                spec.length, f"输入 {spec.name}", color
            )
            handle = TensorHandle(
                name=spec.name,
                storage=storage,
                offset=0,
                length=spec.length,
                role="input",
                color=color,
            )
            self._store(handle, list(spec.values))
            self.handles[spec.name] = handle
            freed = []
            task = self._release_now_if_dead(spec.name)
            if task:
                freed.append(task)
            self._step(
                f"装入输入 {spec.name}",
                f"{spec.length} 个 float32 元素写入缓冲区偏移 {storage.offset}",
                kind="input",
                op=action,
                allocated=[
                    {"name": spec.name, "length": spec.length, "offset": storage.offset}
                ],
                freed=freed,
                compaction=compaction,
            )
            self.current_action = None

    def _run_ops(self) -> None:
        for index, op in enumerate(self.plan.ops):
            action = {
                "name": op.name,
                "kind": op.kind,
                "line": op.line,
                "operands": list(op.operands),
                "length": op.length,
            }
            self.current_action = action
            if op.kind == "slice":
                self._run_slice(index, op, action)
            else:
                self._run_elementwise(index, op, action)
            self.current_action = None

    def _run_slice(self, index: int, op, action: dict) -> None:
        source = self.handles[op.operands[0]]
        assert op.start is not None and op.stop is not None
        view = TensorHandle(
            name=op.name,
            storage=source.storage,
            offset=source.offset + op.start,
            length=op.length,
            role="view",
            color=source.color,
            slice_of=source.name,
        )
        source.storage.refs += 1
        self.handles[op.name] = view
        action["slice"] = {
            "source": source.name,
            "start": op.start,
            "stop": op.stop,
            "source_length": source.length,
            "storage": source.storage.sid,
            "offset": source.storage.offset + view.offset,
        }
        freed = []
        task = self._release_if_done(source.name, index)
        if task:
            freed.append(task)
        task = self._release_now_if_dead(op.name)
        if task:
            freed.append(task)
        if not source.storage.live:
            shared = "来源与视图都已用完，存储块随之回收"
        elif source.released:
            shared = "来源张量已用完，存储块仍由视图保留"
        else:
            shared = "来源张量仍在用，视图与来源共用这块存储"
        self._step(
            f"切片视图 {op.name} = {source.name}[{op.start}:{op.stop}]",
            f"共享 {source.name} 的整块存储（存储 #{source.storage.sid}，"
            f"绝对偏移 {source.storage.offset + view.offset}），不新占元素；{shared}",
            kind="slice",
            op=action,
            view={
                "name": op.name,
                "storage": source.storage.sid,
                "offset": source.storage.offset + view.offset,
                "length": op.length,
                "slice_of": source.name,
            },
            freed=freed,
        )

    def _run_elementwise(self, index: int, op, action: dict) -> None:
        color = self._new_color()
        storage, compaction = self.alloc.allocate(
            op.length, f"算子 {op.name}", color
        )
        handle = TensorHandle(
            name=op.name,
            storage=storage,
            offset=0,
            length=op.length,
            role="result",
            color=color,
        )
        if op.kind == "add":
            left = self._buffered_values(self.handles[op.operands[0]])
            right = self._buffered_values(self.handles[op.operands[1]])
            values = []
            for a, b in zip(left, right):
                result = _round_f32(a + b)
                if math.isinf(result):
                    self.overflow_values += 1
                values.append(result)
            detail = "逐项相加，结果为独立的连续空间"
        else:  # relu
            source_values = self._buffered_values(self.handles[op.operands[0]])
            values = [value if value > 0 else 0.0 for value in source_values]
            detail = "逐项 ReLU（max(0, x)），结果为独立的连续空间"
        self._store(handle, values)
        self.handles[op.name] = handle
        freed = []
        for operand in dict.fromkeys(op.operands):
            task = self._release_if_done(operand, index)
            if task:
                freed.append(task)
        task = self._release_now_if_dead(op.name)
        if task:
            freed.append(task)
        action["sample"] = [fmt_value(value) for value in values[:8]]
        self._step(
            f"执行 {op.name} = {op.kind}({', '.join(op.operands)})",
            detail,
            kind="op",
            op=action,
            allocated=[
                {"name": op.name, "length": op.length, "offset": storage.offset}
            ],
            freed=freed,
            compaction=compaction,
        )

    def _failure(self, exc: OutOfSpace) -> dict:
        attempted = self.current_action or {}
        message = (
            f"需要 {exc.needed} 个连续元素，缓冲区空闲总量只有 {exc.free_total} 个"
            f"（最大连续空闲区 {exc.largest_free} 个），回收后总空闲量仍不足"
        )
        if exc.compactions:
            message += f"；此前已整理 {exc.compactions} 次"
        return {
            "step": len(self.steps) + 1,
            "action": attempted,
            "needed": exc.needed,
            "free_total": exc.free_total,
            "largest_free": exc.largest_free,
            "capacity": self.plan.capacity,
            "compactions": exc.compactions,
            "live": self._live_names(),
            "message": message,
            "layout": self.alloc.layout(self.handles),
        }

    def _finish(self, failure: dict | None) -> dict:
        plan = self.plan
        outputs = []
        for name in plan.outputs:
            handle = self.handles.get(name)
            if handle is None:
                outputs.append(
                    {
                        "name": name,
                        "missing": True,
                        "length": 0,
                        "role": None,
                        "values": [],
                        "storage": None,
                        "offset": None,
                        "slice_of": None,
                    }
                )
                continue
            values = self._buffered_values(handle)
            outputs.append(
                {
                    "name": name,
                    "length": handle.length,
                    "role": handle.role,
                    "color": handle.color,
                    "values": [fmt_value(value) for value in values],
                    "storage": handle.storage.sid,
                    "offset": handle.storage.offset + handle.offset,
                    "slice_of": handle.slice_of,
                }
            )
        peak_percent = round(100.0 * self.alloc.peak / plan.capacity, 2)
        return {
            "ok": failure is None,
            "capacity": plan.capacity,
            "outputs": outputs,
            "failure": failure,
            "steps": self.steps,
            "peak": {
                "elements": self.alloc.peak,
                "percent": peak_percent,
                "at": self.alloc.peak_at,
            },
            "final_occupancy": self.alloc.occupied,
            "final_percent": round(
                100.0 * self.alloc.occupied / plan.capacity, 2
            ),
            "stats": {
                "allocations": self.alloc.allocations,
                "compactions": self.alloc.compactions,
                "moved_elements": self.alloc.moved_elements,
                "steps": len(self.steps),
                "free_total": self.alloc.total_free(),
                "largest_free": self.alloc.largest_free(),
                "overflow_values": self.overflow_values,
            },
            "layout": self.alloc.layout(self.handles),
            "warnings": self.warnings,
        }


def execute_plan(plan: Plan) -> dict:
    executor = Executor(plan)
    try:
        result = executor.run()
    finally:
        executor.alloc.buffer = array("f")
    if result["stats"]["overflow_values"]:
        result["warnings"].append(
            f"{result['stats']['overflow_values']} 个加法结果溢出，按 IEEE-754 记作 inf"
        )
    return result
