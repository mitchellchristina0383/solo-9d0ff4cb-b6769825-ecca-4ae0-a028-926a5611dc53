"""Plan parsing plus static shape / liveness analysis.

A plan is a small text program:

    inputs (one tensor per line, "name: v1, v2, ...")
    ops    (one op per line, "result = add(a, b)" / "relu(x)" / "slice(x, 0, 3)")
    outputs (comma separated tensor names that must stay alive until the end)

All lengths are known statically, so operand compatibility and slice bounds are
validated before a single buffer element is touched.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass

MAX_CAPACITY = 1_000_000
MAX_INPUTS = 64
MAX_OPS = 500
MAX_TENSOR_LEN = 4096
MAX_TOTAL_INPUT_ELEMENTS = MAX_INPUTS * MAX_TENSOR_LEN

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
INPUT_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.*)$")
OP_LINE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$"
)
SPLIT_RE = re.compile(r"[,\s]+")

KINDS = ("add", "relu", "slice")


class ValidationError(ValueError):
    """The submitted plan cannot be parsed or is not semantically valid."""

    def __init__(self, messages: str | list[str]):
        if isinstance(messages, str):
            messages = [messages]
        self.messages = list(messages)
        super().__init__("；".join(self.messages))


def to_float32(value: float) -> float:
    """Round a Python float to the float32 stored in the simulated buffer."""
    if not math.isfinite(value):
        raise ValueError("只接受有限数值（不接受 inf / nan）")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("数值超出 float32 可表示范围") from exc


def fmt_value(value: float) -> str:
    """Compact, deterministic rendering of a float32 value."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    return f"{value:.8g}"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    values: tuple[float, ...]
    line: int = 0

    @property
    def length(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Op:
    name: str
    kind: str
    operands: tuple[str, ...]
    line: int
    start: int | None = None
    stop: int | None = None
    length: int = 0


@dataclass(frozen=True)
class Plan:
    capacity: int
    inputs: tuple[TensorSpec, ...]
    ops: tuple[Op, ...]
    outputs: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def _text(payload: dict, key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"字段 {key} 必须是文本")
    if len(value) > 1_500_000:
        raise ValidationError(f"字段 {key} 过长（上限 1500000 字符）")
    return value


def _logical_lines(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append((lineno, line))
    return out


def _parse_capacity(raw: object, errors: list[str]) -> int | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        errors.append("缓冲容量不能为空（按元素计）")
        return None
    try:
        value = int(text, 10)
    except ValueError:
        errors.append(f"缓冲容量必须是整数，收到 {text!r}")
        return None
    if value < 1:
        errors.append("缓冲容量必须大于 0")
        return None
    if value > MAX_CAPACITY:
        errors.append(f"缓冲容量过大（上限 {MAX_CAPACITY} 个元素）")
        return None
    return value


def _parse_inputs(text: str, errors: list[str]) -> list[TensorSpec]:
    specs: list[TensorSpec] = []
    names: set[str] = set()
    total = 0
    for lineno, line in _logical_lines(text):
        if len(specs) >= MAX_INPUTS:
            errors.append(f"输入张量过多（上限 {MAX_INPUTS} 个）")
            break
        match = INPUT_LINE_RE.match(line)
        if not match:
            errors.append(
                f"输入第 {lineno} 行格式应为“名称: v1, v2, ...”，收到 {line!r}"
            )
            continue
        name, body = match.group(1), match.group(2).strip()
        if name in names:
            errors.append(f"输入第 {lineno} 行：名称 {name} 重复")
            continue
        tokens = [tok for tok in SPLIT_RE.split(body) if tok]
        if not tokens:
            errors.append(f"输入第 {lineno} 行：{name} 至少需要一个数值")
            continue
        if len(tokens) > MAX_TENSOR_LEN:
            errors.append(
                f"输入第 {lineno} 行：{name} 长度 {len(tokens)} 超过上限 {MAX_TENSOR_LEN}"
            )
            continue
        values: list[float] = []
        bad = False
        for token in tokens:
            try:
                values.append(to_float32(float(token)))
            except (ValueError, TypeError):
                errors.append(
                    f"输入第 {lineno} 行：{name} 中的 {token!r} 不是合法的 float32 数值"
                )
                bad = True
                break
        if bad:
            continue
        total += len(values)
        if total > MAX_TOTAL_INPUT_ELEMENTS:
            errors.append(f"输入元素总数超过上限 {MAX_TOTAL_INPUT_ELEMENTS}")
            break
        names.add(name)
        specs.append(TensorSpec(name=name, values=tuple(values), line=lineno))
    if not specs and not errors:
        errors.append("至少需要一个输入张量")
    return specs


def _parse_ops(
    text: str,
    lengths: dict[str, int],
    errors: list[str],
) -> tuple[list[Op], dict[str, int]]:
    ops: list[Op] = []
    known = dict(lengths)
    for lineno, line in _logical_lines(text):
        if len(ops) >= MAX_OPS:
            errors.append(f"算子过多（上限 {MAX_OPS} 个）")
            break
        match = OP_LINE_RE.match(line)
        if not match:
            errors.append(
                f"算子第 {lineno} 行格式应为“结果 = add(a, b) / relu(x) / slice(x, 0, 3)”，收到 {line!r}"
            )
            continue
        name, kind, raw_args = match.group(1), match.group(2), match.group(3)
        if kind not in KINDS:
            errors.append(
                f"算子第 {lineno} 行：未知算子 {kind!r}（支持 add / relu / slice）"
            )
            continue
        if name in known:
            errors.append(f"算子第 {lineno} 行：结果名称 {name} 与已有张量重名")
            continue
        args = [arg.strip() for arg in raw_args.split(",")] if raw_args.strip() else []
        if kind == "add":
            if len(args) != 2:
                errors.append(f"算子第 {lineno} 行：add 需要两个操作数")
                continue
            if any(not NAME_RE.match(arg) for arg in args):
                errors.append(f"算子第 {lineno} 行：add 的操作数必须是张量名称")
                continue
            missing = [arg for arg in args if arg not in known]
            if missing:
                errors.append(
                    f"算子第 {lineno} 行：未定义的张量 {', '.join(missing)}"
                    "（只能引用输入或更早的算子结果）"
                )
                continue
            left, right = (known[args[0]], known[args[1]])
            if left != right:
                errors.append(
                    f"算子第 {lineno} 行：add 要求等长，{args[0]}={left}、{args[1]}={right}"
                )
                continue
            ops.append(Op(name=name, kind=kind, operands=(args[0], args[1]),
                          line=lineno, length=left))
            known[name] = left
        elif kind == "relu":
            if len(args) != 1:
                errors.append(f"算子第 {lineno} 行：relu 需要一个操作数")
                continue
            if not NAME_RE.match(args[0]):
                errors.append(f"算子第 {lineno} 行：relu 的操作数必须是张量名称")
                continue
            if args[0] not in known:
                errors.append(f"算子第 {lineno} 行：未定义的张量 {args[0]}")
                continue
            ops.append(Op(name=name, kind=kind, operands=(args[0],),
                          line=lineno, length=known[args[0]]))
            known[name] = known[args[0]]
        else:  # slice
            if len(args) not in (3, 4):
                errors.append(
                    f"算子第 {lineno} 行：slice 需要“源张量, 起始, 结束”（可选步长，且必须为 1）"
                )
                continue
            src = args[0]
            if not NAME_RE.match(src) or src not in known:
                errors.append(f"算子第 {lineno} 行：未定义的切片源张量 {src!r}")
                continue
            try:
                start_raw = None if args[1] == "" else int(args[1], 10)
                stop_raw = None if args[2] == "" else int(args[2], 10)
                step = 1 if len(args) == 3 or args[3] == "" else int(args[3], 10)
            except ValueError:
                errors.append(f"算子第 {lineno} 行：切片参数必须是整数")
                continue
            if step != 1:
                errors.append(
                    f"算子第 {lineno} 行：连续切片要求步长为 1，收到 {step}"
                )
                continue
            src_len = known[src]
            start, stop, _ = slice(start_raw, stop_raw, 1).indices(src_len)
            new_len = stop - start
            if new_len <= 0:
                errors.append(
                    f"算子第 {lineno} 行：切片 {src}[{start}:{stop}] 为空（源长 {src_len}）"
                )
                continue
            ops.append(
                Op(name=name, kind=kind, operands=(src,), line=lineno,
                   start=start, stop=stop, length=new_len)
            )
            known[name] = new_len
    return ops, known


def _parse_outputs(
    text: str, known: dict[str, int], errors: list[str]
) -> tuple[str, ...]:
    outputs: list[str] = []
    for token in (tok for tok in SPLIT_RE.split(text.strip()) if tok):
        if not NAME_RE.match(token):
            errors.append(f"输出名称 {token!r} 不合法")
            continue
        if token not in known:
            errors.append(f"输出 {token} 未定义")
            continue
        if token not in outputs:
            outputs.append(token)
    if not outputs:
        errors.append("至少需要保留一个输出张量")
    return tuple(outputs)


def _collect_warnings(
    inputs: list[TensorSpec], ops: list[Op], outputs: tuple[str, ...]
) -> tuple[str, ...]:
    used = {name for op in ops for name in op.operands}
    output_set = set(outputs)
    warnings: list[str] = []
    for spec in inputs:
        if spec.name not in used and spec.name not in output_set:
            warnings.append(f"输入 {spec.name} 未被使用，分配后会立即回收")
    for op in ops:
        if op.name not in used and op.name not in output_set:
            warnings.append(f"中间结果 {op.name} 未被使用，产生后会立即回收")
    return tuple(warnings)


def parse_plan(payload: dict) -> Plan:
    """Validate *payload* and return an immutable Plan (or raise ValidationError)."""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    errors: list[str] = []
    capacity = _parse_capacity(payload.get("capacity"), errors)
    inputs_text = _text(payload, "inputs")
    ops_text = _text(payload, "ops")
    outputs_text = _text(payload, "outputs")
    inputs = _parse_inputs(inputs_text, errors)
    lengths = {spec.name: spec.length for spec in inputs}
    ops, known = _parse_ops(ops_text, lengths, errors)
    outputs = _parse_outputs(outputs_text, known, errors)
    if errors:
        raise ValidationError(errors)
    assert capacity is not None
    for name, length in known.items():
        if length > MAX_CAPACITY:
            raise ValidationError(f"张量 {name} 的长度 {length} 超出容量上限")
    warnings = _collect_warnings(list(inputs), ops, outputs)
    return Plan(
        capacity=capacity,
        inputs=tuple(inputs),
        ops=tuple(ops),
        outputs=outputs,
        warnings=warnings,
    )
