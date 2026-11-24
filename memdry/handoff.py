"""Split-run handoff package: build at the boundary and load on the second device.

A handoff package is a self-contained JSON document carrying:

* ``back_ops``  -- the part of the plan that still has to run;
* ``storages``  -- every *source storage block* that is alive at the boundary and
  needed (directly or through a view) by a back-segment op or retained output.

Only the **visible elements** of each block are serialized: the union of the
relative spans of the live tensors sharing that block, so overlapping slices
cost their union rather than their sum, and each storage position is sent at
most once.  Values are packed as little-endian IEEE-754 binary32 and base64
encoded, which preserves the exact float32 bit pattern (including ``-0`` and
``inf``).

On the receiving side the loader validates the document statically and turns
it into :class:`LoadedStorage` objects that :class:`BackExecutor` can install.
"""

from __future__ import annotations

import base64
import binascii
import struct
from array import array
from dataclasses import dataclass

from .memory import Executor, OutOfSpace, TensorHandle
from .model import (
    KINDS,
    NAME_RE,
    MAX_CAPACITY,
    MAX_OPS,
    Op,
    Plan,
    ValidationError,
)

HANDOFF_FORMAT = "memdry-handoff/1"


@dataclass(frozen=True)
class LoadedTensor:
    name: str
    role: str
    rel_offset: int
    length: int
    color: int
    slice_of: str | None
    released: bool = False


@dataclass(frozen=True)
class LoadedStorage:
    origin_sid: int
    length: int
    label: str
    color: int
    # (relative start, exact float32 values) per non-overlapping segment
    segments: tuple[tuple[int, tuple[float, ...]], ...]
    tensors: tuple[LoadedTensor, ...]


@dataclass(frozen=True)
class Handoff:
    split_index: int
    split_op: dict
    ops: tuple[Op, ...]
    outputs: tuple[str, ...]
    storages: tuple[LoadedStorage, ...]
    transferred_elements: int


# -- serialization -----------------------------------------------------------

def _op_dict(op: Op) -> dict:
    return {
        "name": op.name,
        "kind": op.kind,
        "operands": list(op.operands),
        "line": op.line,
        "start": op.start,
        "stop": op.stop,
        "length": op.length,
    }


def _pack_values(values) -> str:
    # ``values`` is an array('f') slice; tobytes() keeps the exact binary32 bits.
    return base64.b64encode(array("f", values).tobytes()).decode("ascii")


def build_handoff(executor: Executor, split_index: int) -> dict:
    """Snapshot the boundary state of a front-segment *executor*."""
    plan = executor.plan
    back_ops = plan.ops[split_index + 1 :]
    back_defined = {op.name for op in back_ops}

    required: set[str] = set()
    for op in back_ops:
        for operand in op.operands:
            if operand not in back_defined:
                required.add(operand)
    for name in plan.outputs:
        if name not in back_defined:
            required.add(name)

    groups: dict[int, list[TensorHandle]] = {}

    def include(name: str) -> None:
        handle = executor.handles.get(name)
        # A released source handle is fine as long as its storage is still
        # alive (a live view keeps the whole block around).
        if handle is None or not handle.storage.live:
            raise ValidationError(f"分界处张量 {name} 的存储已回收，无法交接")
        bucket = groups.setdefault(handle.storage.sid, [])
        if all(item.name != name for item in bucket):
            bucket.append(handle)
            # A view names its source tensor; carry the source too so the
            # back segment can reconstruct the slice relationship. It shares
            # the same storage, so this may add no extra visible elements.
            if handle.slice_of is not None:
                include(handle.slice_of)

    for name in sorted(required):
        include(name)

    storages: list[dict] = []
    transferred = 0
    for sid in sorted(groups):
        handles = sorted(groups[sid], key=lambda h: h.name)
        storage = handles[0].storage
        # Merge the relative spans of *live* tensors: overlapping slices
        # transfer their union, and every storage position is serialized at
        # most once. Released handles (e.g. a view's already-dead source) are
        # carried only as relationship metadata and add no visible elements.
        live_handles = [h for h in handles if not h.released]
        intervals: list[list[int]] = []
        for handle in sorted((h.offset, h.offset + h.length) for h in live_handles):
            start, end = handle
            if intervals and start <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])

        segments = []
        for rel_start, rel_end in intervals:
            length = rel_end - rel_start
            absolute = storage.offset + rel_start
            values = executor.alloc.buffer[absolute : absolute + length]
            segments.append(
                {"start": rel_start, "length": length, "values_b64": _pack_values(values)}
            )
            transferred += length

        tensors = [
            {
                "name": h.name,
                "role": h.role,
                "rel_offset": h.offset,
                "length": h.length,
                "slice_of": h.slice_of,
                "color": h.color,
                "released": h.released,
            }
            for h in handles
        ]
        storages.append(
            {
                "origin_sid": sid,
                "length": storage.length,
                "label": storage.label,
                "color": storage.color,
                "segments": segments,
                "tensors": tensors,
            }
        )

    return {
        "format": HANDOFF_FORMAT,
        "split_after": split_index,
        "split_op": _op_dict(plan.ops[split_index]),
        "back_ops": [_op_dict(op) for op in back_ops],
        "outputs": list(plan.outputs),
        "storages": storages,
        "transferred_elements": transferred,
    }


# -- loading / validation ----------------------------------------------------

def _as_int(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{what} 必须是整数")
    return value


def _decode_segment(raw_segment: dict, storage_length: int) -> tuple[int, tuple[float, ...]]:
    if not isinstance(raw_segment, dict):
        raise ValidationError("交接包的 segments 项必须是对象")
    start = _as_int(raw_segment.get("start"), "分段起始")
    length = _as_int(raw_segment.get("length"), "分段长度")
    if start < 0 or length < 1 or start + length > storage_length:
        raise ValidationError(
            f"分段越界：start={start}, length={length}，存储块长度 {storage_length}"
        )
    encoded = raw_segment.get("values_b64")
    if not isinstance(encoded, str):
        raise ValidationError("分段缺少 values_b64（base64 编码的 float32 数据）")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("分段 values_b64 不是合法的 base64") from exc
    if len(raw) % 4:
        raise ValidationError("分段二进制长度不是 4 的倍数，无法按 float32 解码")
    count = len(raw) // 4
    if count != length:
        raise ValidationError(
            f"分段声明 {length} 个元素，base64 解码后实际为 {count} 个 float32"
        )
    try:
        values = struct.unpack("<%df" % count, raw)
    except struct.error as exc:
        raise ValidationError("分段 float32 解码失败") from exc
    return start, values


def _load_op(raw: dict, known: dict[str, int], seen_names: set[str]) -> Op:
    if not isinstance(raw, dict):
        raise ValidationError("后段算子必须是对象")
    name = raw.get("name")
    kind = raw.get("kind")
    operands = raw.get("operands")
    line = raw.get("line", 0)
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValidationError(f"算子结果名 {name!r} 不合法")
    if name in seen_names:
        raise ValidationError(f"后段算子结果名 {name} 重复")
    if kind not in KINDS:
        raise ValidationError(f"算子 {name} 的类型 {kind!r} 不支持（仅 add / relu / slice）")
    if not isinstance(operands, list) or not all(
        isinstance(item, str) and NAME_RE.match(item) for item in operands
    ):
        raise ValidationError(f"算子 {name} 的 operands 必须是名称数组")
    if not isinstance(line, int) or isinstance(line, bool) or line < 0:
        raise ValidationError(f"算子 {name} 的行号不合法")

    def require_known(item: str) -> int:
        if item not in known:
            raise ValidationError(
                f"算子 {name} 引用了未定义的张量 {item}（既不在交接包中，也不是更早的后段结果）"
            )
        return known[item]

    if kind == "add":
        if len(operands) != 2:
            raise ValidationError(f"算子 {name}：add 需要两个操作数")
        left = require_known(operands[0])
        right = require_known(operands[1])
        if left != right:
            raise ValidationError(f"算子 {name}：add 要求等长，{operands[0]}={left}、{operands[1]}={right}")
        op = Op(name=name, kind=kind, operands=tuple(operands), line=line, length=left)
    elif kind == "relu":
        if len(operands) != 1:
            raise ValidationError(f"算子 {name}：relu 需要一个操作数")
        length = require_known(operands[0])
        op = Op(name=name, kind=kind, operands=tuple(operands), line=line, length=length)
    else:
        if len(operands) != 1:
            raise ValidationError(f"算子 {name}：slice 需要一个源张量")
        src_len = require_known(operands[0])
        start = _as_int(raw.get("start"), f"算子 {name} 的切片起点")
        stop = _as_int(raw.get("stop"), f"算子 {name} 的切片终点")
        if start is None or stop is None or not (0 <= start < stop <= src_len):
            raise ValidationError(
                f"算子 {name}：切片 [{start}:{stop}] 对源长 {src_len} 不合法"
            )
        op = Op(
            name=name,
            kind=kind,
            operands=tuple(operands),
            line=line,
            start=start,
            stop=stop,
            length=stop - start,
        )
    seen_names.add(name)
    known[name] = op.length
    return op


def load_handoff(data: object) -> Handoff:
    """Validate a handoff document and return a runnable :class:`Handoff`."""
    if not isinstance(data, dict):
        raise ValidationError("交接包必须是 JSON 对象")
    if data.get("format") != HANDOFF_FORMAT:
        raise ValidationError(
            f"交接包格式标识不匹配（期望 {HANDOFF_FORMAT}，收到 {data.get('format')!r}）"
        )

    split_index = _as_int(data.get("split_after"), "split_after")
    if split_index < 0:
        raise ValidationError("split_after 不能为负数")
    split_op = data.get("split_op")
    if not isinstance(split_op, dict) or "name" not in split_op:
        raise ValidationError("交接包缺少分界算子信息 split_op")

    raw_ops = data.get("back_ops")
    if not isinstance(raw_ops, list) or not raw_ops:
        raise ValidationError("交接包的后段计划 back_ops 必须是非空数组")
    if len(raw_ops) > MAX_OPS:
        raise ValidationError(f"后段算子过多（上限 {MAX_OPS} 个）")

    raw_storages = data.get("storages")
    if not isinstance(raw_storages, list) or not raw_storages:
        raise ValidationError("交接包缺少分界处的存储块 storages")

    outputs_raw = data.get("outputs")
    if not isinstance(outputs_raw, list) or not outputs_raw:
        raise ValidationError("交接包缺少要保留的输出 outputs")
    outputs: list[str] = []
    for name in outputs_raw:
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ValidationError(f"输出名称 {name!r} 不合法")
        if name not in outputs:
            outputs.append(name)

    storages: list[LoadedStorage] = []
    known: dict[str, int] = {}
    seen_tensors: set[str] = set()
    total_elements = 0

    for index, raw_storage in enumerate(raw_storages):
        if not isinstance(raw_storage, dict):
            raise ValidationError(f"存储块 #{index} 必须是对象")
        origin_sid = _as_int(raw_storage.get("origin_sid"), f"存储块 #{index} 的 origin_sid")
        if origin_sid < 0:
            raise ValidationError("origin_sid 不能为负数")
        length = _as_int(raw_storage.get("length"), f"存储块 #{index} 的 length")
        if length < 1 or length > MAX_CAPACITY:
            raise ValidationError(f"存储块 #{index} 长度 {length} 不合法")
        label = raw_storage.get("label", f"来源存储 #{origin_sid}")
        if not isinstance(label, str):
            raise ValidationError(f"存储块 #{index} 的 label 必须是文本")
        color = raw_storage.get("color", 0)
        if not isinstance(color, int) or isinstance(color, bool):
            raise ValidationError(f"存储块 #{index} 的 color 必须是整数")
        total_elements += length

        raw_segments = raw_storage.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValidationError(f"存储块 #{index} 至少需要一个可见元素分段")
        segments = [_decode_segment(item, length) for item in raw_segments]
        segments.sort(key=lambda item: item[0])
        for (s1, v1), (s2, _) in zip(segments, segments[1:]):
            if s2 < s1 + len(v1):
                raise ValidationError(f"存储块 #{index} 的分段存在重叠，同一位置被重复传输")

        raw_tensors = raw_storage.get("tensors")
        if not isinstance(raw_tensors, list) or not raw_tensors:
            raise ValidationError(f"存储块 #{index} 至少需要一个张量")
        tensors: list[LoadedTensor] = []
        block_names: set[str] = set()
        for raw_tensor in raw_tensors:
            if not isinstance(raw_tensor, dict):
                raise ValidationError(f"存储块 #{index} 的张量项必须是对象")
            tname = raw_tensor.get("name")
            if not isinstance(tname, str) or not NAME_RE.match(tname):
                raise ValidationError(f"存储块 #{index} 中张量名 {tname!r} 不合法")
            if tname in block_names:
                raise ValidationError(f"张量 {tname} 在同一存储块中重复出现")
            role = raw_tensor.get("role")
            if role not in ("input", "result", "view"):
                raise ValidationError(f"张量 {tname} 的 role {role!r} 不合法")
            rel_offset = _as_int(raw_tensor.get("rel_offset"), f"张量 {tname} 的 rel_offset")
            tlength = _as_int(raw_tensor.get("length"), f"张量 {tname} 的 length")
            if rel_offset < 0 or tlength < 1 or rel_offset + tlength > length:
                raise ValidationError(
                    f"张量 {tname} 的区间 [{rel_offset}:{rel_offset + tlength}] 超出存储块长度 {length}"
                )
            tcolor = raw_tensor.get("color", 0)
            if not isinstance(tcolor, int) or isinstance(tcolor, bool):
                raise ValidationError(f"张量 {tname} 的 color 必须是整数")
            slice_of = raw_tensor.get("slice_of")
            if role == "view":
                if not isinstance(slice_of, str) or not NAME_RE.match(slice_of):
                    raise ValidationError(f"切片视图 {tname} 缺少合法的 slice_of")
            elif slice_of is not None:
                raise ValidationError(f"张量 {tname} 不是视图，不应携带 slice_of")
            released = raw_tensor.get("released", False)
            if not isinstance(released, bool):
                raise ValidationError(f"张量 {tname} 的 released 必须是布尔值")
            if not released:
                # Every visible element of a live tensor must lie inside a
                # transferred segment.
                covered = [
                    s <= rel_offset and s + len(v) >= rel_offset + tlength
                    for s, v in segments
                ]
                if not any(covered):
                    raise ValidationError(
                        f"张量 {tname} 引用了未传输的元素（分段只覆盖可见元素）"
                    )
            tensors.append(
                LoadedTensor(
                    name=tname,
                    role=role,
                    rel_offset=rel_offset,
                    length=tlength,
                    color=tcolor,
                    slice_of=slice_of,
                    released=released,
                )
            )
            block_names.add(tname)
            if not released:
                if tname in seen_tensors:
                    raise ValidationError(f"张量 {tname} 在交接包中重复出现")
                seen_tensors.add(tname)
                known[tname] = tlength

        # Views must point at another tensor in the same source block.
        local_names = {tensor.name for tensor in tensors}
        for tensor in tensors:
            if tensor.role == "view" and tensor.slice_of not in local_names:
                raise ValidationError(
                    f"视图 {tensor.name} 的来源 {tensor.slice_of} 不在同一来源存储块中"
                )
        storages.append(
            LoadedStorage(
                origin_sid=origin_sid,
                length=length,
                label=label,
                color=color,
                segments=tuple(segments),
                tensors=tuple(tensors),
            )
        )

    if total_elements > MAX_CAPACITY:
        raise ValidationError(f"交接存储元素总数超过上限 {MAX_CAPACITY}")

    seen_op_names = set(known)
    ops = tuple(_load_op(raw, known, seen_op_names) for raw in raw_ops)

    for name in outputs:
        if name not in known:
            raise ValidationError(f"保留输出 {name} 既未在交接包中，也不是后段算子结果")

    # A transferred tensor that nobody consumes would inflate the back peak;
    # the front side only sends needed tensors (plus same-block view sources),
    # so reject drift explicitly.
    back_operands = {operand for op in ops for operand in op.operands}
    view_sources = {
        tensor.slice_of
        for storage in storages
        for tensor in storage.tensors
        if tensor.role == "view"
    }
    for tensor_name in seen_tensors:
        if (
            tensor_name not in back_operands
            and tensor_name not in outputs
            and tensor_name not in view_sources
        ):
            raise ValidationError(f"交接张量 {tensor_name} 未被后段算子或保留输出使用")

    transferred = sum(len(values) for storage in storages for _, values in storage.segments)
    return Handoff(
        split_index=split_index,
        split_op=split_op,
        ops=ops,
        outputs=tuple(outputs),
        storages=tuple(storages),
        transferred_elements=transferred,
    )


# -- back-segment execution --------------------------------------------------

class BackExecutor(Executor):
    """Resume a plan from a handoff package inside a fresh fixed buffer."""

    def __init__(self, handoff: Handoff, capacity: int):
        plan = Plan(
            capacity=capacity,
            inputs=(),
            ops=handoff.ops,
            outputs=handoff.outputs,
            warnings=(),
        )
        super().__init__(plan)
        self.handoff = handoff
        self.phase = "load"

    def run(self) -> dict:
        try:
            self._load_storages()
            self.phase = "op"
            self._run_ops()
        except OutOfSpace as exc:
            failure = self._failure(exc)
            failure["phase"] = self.phase
            failure["phase_label"] = (
                "载入交接包" if self.phase == "load" else "执行后段算子"
            )
            return self._finish(failure)
        return self._finish(None)

    def _load_storages(self) -> None:
        for loaded in self.handoff.storages:
            self.current_action = {
                "name": loaded.label,
                "kind": "load",
                "line": 0,
                "operands": [],
                "length": loaded.length,
                "origin_sid": loaded.origin_sid,
            }
            # Each source storage is allocated once at its original length and
            # shared by the source tensor and all same-origin slices.
            storage, compaction = self.alloc.allocate(
                loaded.length, loaded.label, loaded.color
            )
            for start, values in loaded.segments:
                absolute = storage.offset + start
                self.alloc.buffer[absolute : absolute + len(values)] = array(
                    "f", values
                )
            live_tensors = [tensor for tensor in loaded.tensors if not tensor.released]
            # One reference per live tensor; released view sources are kept
            # only as metadata and hold no reference.
            storage.refs = len(live_tensors)
            for tensor in live_tensors:
                self.handles[tensor.name] = TensorHandle(
                    name=tensor.name,
                    storage=storage,
                    offset=tensor.rel_offset,
                    length=tensor.length,
                    role=tensor.role,
                    color=tensor.color,
                    slice_of=tensor.slice_of,
                )
            self._step(
                f"载入来源存储 #{loaded.origin_sid}（{loaded.label}）",
                f"按原长 {loaded.length} 个元素分配一次，写入 "
                f"{sum(len(v) for _, v in loaded.segments)} 个可见元素，"
                f"供 {len(live_tensors)} 个存活张量（含同源切片）共用",
                kind="load",
                op=dict(self.current_action),
                allocated=[
                    {"name": loaded.label, "length": loaded.length, "offset": storage.offset}
                ],
                compaction=compaction,
            )
            self.current_action = None

    def _failure(self, exc: OutOfSpace) -> dict:
        failure = super()._failure(exc)
        failure["phase"] = self.phase
        failure["phase_label"] = (
            "载入交接包" if self.phase == "load" else "执行后段算子"
        )
        return failure


def execute_handoff(handoff: Handoff, capacity: int) -> dict:
    executor = BackExecutor(handoff, capacity)
    try:
        result = executor.run()
    finally:
        executor.alloc.buffer = array("f")
    result["handoff_meta"] = {
        "split_after": handoff.split_index,
        "split_op": handoff.split_op.get("name"),
        "transferred_elements": handoff.transferred_elements,
        "storage_count": len(handoff.storages),
    }
    if result["stats"]["overflow_values"]:
        result["warnings"].append(
            f"{result['stats']['overflow_values']} 个加法结果溢出，按 IEEE-754 记作 inf"
        )
    return result
