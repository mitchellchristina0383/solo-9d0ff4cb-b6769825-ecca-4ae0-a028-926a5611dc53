# 端侧推理 · 内存试跑（Flask）

把一份推理计划（输入张量、算子顺序、要保留的输出、缓冲容量）放进**一块定长
float32 缓冲**里模拟执行，观察每一步的分配、引用计数回收、碎片整理与占用峰值。

- 输入、中间值、输出全部驻留同一块缓冲区，搬移（compaction）也在其中完成。
- `add`、`relu`：为结果分配**独立的连续空间**。
- `slice`：**共享来源张量的整块存储**的视图，不额外占用元素。
- 引用计数：来源张量与视图都释放后，整块存储才回收；指定输出保留到执行结束。
- 分配器按“首次适配 + 空闲区合并”：若单块空闲不够但**总空闲量足够**，就把存活
  块向缓冲区头部紧凑搬移（视图随所属存储一起移动，相对偏移不变）后重试。
- 容量确实不够时返回**失败位置、所需空间、当时总/最大空闲量与仍在使用的值**。
- 成功时返回各输出值、缓冲占用峰值与逐步时间线。

## 运行（Docker Compose）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

`MEMDRY_PORT` 可改宿主机端口：`MEMDRY_PORT=8080 docker compose up --build`。

服务用 gunicorn（2 worker × 4 线程）跑 `memdry.app:app`，容器只读、去权、带
healthcheck（`GET /healthz`）。镜像基于 `python:3.12-slim`，依赖只有 Flask 与
gunicorn（`requirements.txt` 已锁版本）。

> 本机验证环境未安装 Docker（`docker` 命令不存在），因此 compose 文件与 Dockerfile
> 只做了静态校验和同参数本地 gunicorn 试跑，详见“实测结果 / 未完成项”。

## 本机直接跑（等价路径，用于验证）

```bash
python3 -m pip install -r requirements.txt
python3 -m gunicorn --bind 127.0.0.1:8000 --workers 2 --threads 4 \
  --no-control-socket --worker-tmp-dir /tmp memdry.app:app
# 或开发模式：PORT=8000 python3 -m memdry.app
```

## 输入格式

页面左侧四个字段，对应 `POST /api/run` 的 JSON：

| 字段 | 含义 |
| --- | --- |
| `capacity` | 缓冲容量，按 float32 元素计（1 ≤ n ≤ 1,000,000） |
| `inputs` | 每行 `名称: v1, v2, ...`，逗号或空格分隔 |
| `ops` | 每行一个算子：`r = add(a, b)` / `r = relu(x)` / `v = slice(x, 起始, 结束)` |
| `outputs` | 逗号分隔的输出张量名，驻留到执行结束 |

规则：`add` 要求两操作数等长；`slice` 必须是步长 1 的连续切片，下标语义与 Python
切片一致（支持负数、省略端点，结果不能为空）；只能引用输入或更早的算子结果。

```json
{
  "capacity": "24",
  "inputs": "X: 1, -2, 3, -4\nA: 10, -20, 30, -40, 50, -60\nY: -1, -2, 6, 7, -8, 9",
  "ops": "v = slice(A, 1, 4)\np = relu(X)\nq = relu(Y)",
  "outputs": "v, p, q"
}
```

## 页面上的三个示例

- **空闲区分散 → 整理**：容量 24，四个 relu 结果与输入交替释放，`w` 产生前只剩
  4+4 两段空闲；整理后最大连续空闲 4 → 8，分配成功。
- **保留切片视图**：`A` 的切片 `v` 在 `A` 释放后仍有效；随后 `q` 的结果分配触发
  整理，`A` 的整块存储连同视图从偏移 4 搬到偏移 0，输出值保持不变。
- **容量不足**：容量 10 时 `c = add(a, b)` 需要 4 个连续元素、空闲只有 2 个，
  页面给出失败步骤（第 3 步）、行号、所需/可用空间与仍占用的 `a, b`。

## 接口

`POST /api/run` → 200 + 报告 JSON（容量不足也是 200，`ok=false` 并带 `failure`；
格式/语义错误返回 400 + `errors`）。

主要字段：`ok`、`outputs[]`（值、角色 `input/result/view`、存储号与偏移、`slice_of`）、
`peak {elements, percent, at}`、`final_occupancy`、`stats {allocations, compactions,
moved_elements, free_total, largest_free, overflow_values}`、`steps[]`（每步的
`allocated / freed / view / compaction / occupancy`）、`layout[]`（当前存储与空闲区，
含视图相对区间）、`failure`、`warnings`。

## 代码结构

```
memdry/model.py    计划解析 + 静态校验（长度、切片边界、名称、上限）
memdry/memory.py   定长缓冲分配器（首次适配、合并、整理）+ 执行器（引用计数、视图）
memdry/app.py      Flask 路由：/、/healthz、POST /api/run
templates/index.html, static/{app.js,style.css}   单页界面
tests/             pytest：分配器、执行器、解析与 HTTP
```

## 测试

```bash
python3 -m pip install pytest
python3 -m pytest -q          # 31 passed
```

## 实测结果

本机 Python 3.11 + Flask 3.1.3 + gunicorn 27.0.1。

| 场景 | 结果 |
| --- | --- |
| `pytest` | 31 passed |
| 空闲区分散示例 | 峰值 22/24（91.67%），整理 1 次、搬移 10 元素，最大连续空闲 4→8，四个输出正确 |
| 保留切片 + 整理 | 峰值 22/24，整理 1 次、搬移 16 元素，视图从偏移 5 移到 1，值 `-20, 30, -40` 不变 |
| 容量不足 | `ok=false`，失败于第 3 步 `c = add(a, b)`，需要 4、空闲 2、最大连续 2，`live=[a, b]` |
| 视图共享/引用计数 | `v=slice(x,1,4)` 时 `x` 释放但存储保留；视图释放后才归还整块 |
| 随机对照（2977 个有效计划） | 与无缓冲参考实现的逐元素结果一致；布局无重叠、占用/空闲守恒 |
| 分配器随机压测（500 轮 × 80 次操作） | 搬移后内容不变、无重叠、失败仅当总空闲不足 |
| gunicorn 双 worker 试跑 | `/healthz` 200；页面与两个静态资源 200；API 成功/失败路径正常 |

## 未完成项与已知限制

- **未在真实 Docker 中构建运行**：当前环境没有 `docker`/`podman`。已按 compose
  参数（2 worker、`--no-control-socket`、只读根文件系统、tmpfs `/tmp`）在本地
  gunicorn 上跑通，并对 YAML 做了解析校验；首次 `docker compose up --build` 仍需
  在有 Docker 的机器上确认。
- **未做浏览器端视觉核对**：环境没有 Chromium/Firefox。前端用 jsdom 驱动真实接口
  做了 DOM 级冒烟测试（三个示例 + 校验错误），但没有渲染截图。
- 模拟只覆盖 `add`、`relu`、步长 1 的连续切片；不支持广播、跨步/负步长切片、
  多输出算子、非 float32 类型。
- 加法按 `array('f')` 的 IEEE-754 binary32 做逐项运算；极少数情况下中间舍入与硬件
  FMA 的实现可能相差 1 ulp。溢出记作 `inf` 并在提示中计数。
- 分配策略是确定性的首次适配＋全量整理；不模拟真实硬件对齐、内存池或分页开销，
  峰值统计的是元素数而非字节。
- 并发/多用户未做隔离与限流；服务端每次请求独立计算，无状态。
