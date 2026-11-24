/* Memory dry-run page logic: submit plan, render report, split handoff. */
(function () {
  "use strict";

  var PRESETS = {
    fragmentation: {
      capacity: "24",
      inputs: "a: 1, -4, 3, 2\nsep: 5, 5\nb: -1, 6, 7, -2\nc: 2, -3, 4, 8, -5, 1",
      ops: "u = relu(a)\nv = relu(b)\nw = relu(c)\nz = relu(sep)",
      outputs: "u, v, w, z",
      split_after: "",
    },
    slice: {
      capacity: "24",
      inputs: "X: 1, -2, 3, -4\nA: 10, -20, 30, -40, 50, -60\nY: -1, -2, 6, 7, -8, 9",
      ops: "v = slice(A, 1, 4)\np = relu(X)\nq = relu(Y)",
      outputs: "v, p, q",
      split_after: "",
    },
    full: {
      capacity: "10",
      inputs: "a: 1, 2, 3, 4\nb: 10, 20, 30, 40",
      ops: "c = add(a, b)",
      outputs: "c",
      split_after: "",
    },
    overlap: {
      capacity: "16",
      inputs: "X: 1, -2, 3, -4, 5, -6, 7, -8\nY: 10, 20, 30, 40",
      ops: "v1 = slice(X, 1, 5)\nv2 = slice(X, 3, 7)\nr = relu(v1)\ns = add(v2, Y)",
      outputs: "r, s",
      split_after: "1",
    },
  };

  var PALETTE = [
    "#58d3c4", "#7c9cff", "#f2b544", "#ff8fab",
    "#9ae66e", "#c79bff", "#f4845f", "#6fd3f7",
  ];

  var $ = function (id) { return document.getElementById(id); };
  var selectedPackage = null;
  var selectedPackageName = "";

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function fmt(n) {
    return Number(n).toLocaleString("zh-CN");
  }

  function colorFor(i) {
    return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];
  }

  // -- form ---------------------------------------------------------------
  function loadPreset(name) {
    var preset = PRESETS[name];
    if (!preset) return;
    $("capacity").value = preset.capacity;
    $("inputs").value = preset.inputs;
    $("ops").value = preset.ops;
    $("outputs").value = preset.outputs;
    syncSplitOptions(preset.split_after);
    clearErrors();
    setStatus("");
  }

  // Parse op result names in text order so the dropdown tracks the textarea.
  function opNames() {
    var names = [];
    $("ops").value.split(/\r?\n/).forEach(function (line) {
      var match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=/);
      if (match) names.push(match[1]);
    });
    return names;
  }

  function syncSplitOptions(preferred) {
    var select = $("split-after");
    var previous = preferred !== undefined ? String(preferred) : select.value;
    var names = opNames();
    select.innerHTML = "";
    select.appendChild(new Option("不分段：整份计划一次跑完", ""));
    names.forEach(function (name, index) {
      var disabled = index === names.length - 1;
      var option = new Option(
        (index + 1) + ". " + name + (disabled ? "（最后一个算子，后段为空不可选）" : ""),
        String(index)
      );
      option.disabled = disabled;
      select.appendChild(option);
    });
    // Keep the previous choice when it is still valid; a split at the last
    // op would leave the back segment empty, so fall back to the one before.
    var valid = Array.prototype.map.call(select.options, function (o) { return o.value; });
    if (valid.indexOf(previous) >= 0 && previous !== String(names.length - 1)) {
      select.value = previous;
    } else if (names.length > 1) {
      select.value = String(Math.max(0, names.length - 2));
    } else {
      select.value = "";
    }
  }

  function setStatus(text) {
    $("run-status").textContent = text;
  }

  function setBackStatus(text) {
    $("back-status").textContent = text;
  }

  function clearErrors() {
    var box = $("form-errors");
    box.hidden = true;
    box.innerHTML = "";
  }

  function showErrors(messages) {
    var box = $("form-errors");
    box.hidden = false;
    box.innerHTML = "";
    var list = el("ul");
    messages.forEach(function (message) {
      list.appendChild(el("li", null, message));
    });
    box.appendChild(list);
  }

  function showBackErrors(messages) {
    var box = $("back-errors");
    box.hidden = false;
    box.innerHTML = "";
    var list = el("ul");
    messages.forEach(function (message) {
      list.appendChild(el("li", null, message));
    });
    box.appendChild(list);
  }

  function clearBackErrors() {
    var box = $("back-errors");
    box.hidden = true;
    box.innerHTML = "";
  }

  function payload() {
    return {
      capacity: $("capacity").value.trim(),
      inputs: $("inputs").value,
      ops: $("ops").value,
      outputs: $("outputs").value,
      split_after: $("split-after").value,
    };
  }

  function setBusy(busy) {
    var button = $("run");
    button.disabled = busy;
    button.textContent = busy ? "试跑中…" : "开始试跑";
  }

  async function run() {
    clearErrors();
    setBusy(true);
    setStatus("正在按算子顺序模拟缓冲区…");
    $("empty").hidden = true;
    try {
      var response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload()),
      });
      var data = await response.json().catch(function () { return null; });
      if (!response.ok || !data) {
        var messages =
          data && data.errors && data.errors.length
            ? data.errors
            : ["请求失败（HTTP " + response.status + "），请稍后重试"];
        showErrors(messages);
        $("result").hidden = true;
        $("empty").hidden = false;
        setStatus("");
        return;
      }
      render(data);
      setStatus(data.handoff ? "前段试跑完成，交接包已就绪" : "");
    } catch (err) {
      showErrors(["无法连接到服务器：" + err.message]);
      $("empty").hidden = false;
      $("result").hidden = true;
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  // -- handoff package ----------------------------------------------------
  function downloadPackage(pkg) {
    var blob = new Blob([JSON.stringify(pkg, null, 2)], {
      type: "application/json",
    });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "memdry-handoff-" + pkg.split_op.name + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function renderHandoff(data) {
    var block = $("handoff-block");
    if (!data.handoff) {
      block.hidden = true;
      return;
    }
    block.hidden = false;
    var h = data.handoff;
    var box = $("handoff-summary");
    box.innerHTML = "";
    function row(key, value) {
      var r = el("div", "row");
      r.appendChild(el("span", "k", key));
      r.appendChild(el("span", "v", value));
      box.appendChild(r);
    }
    row("分界算子", h.split_op.name + "（第 " + (h.split_after + 1) + " 个，后段剩 " +
      h.package.back_ops.length + " 个算子）");
    row("传输元素数", fmt(h.transferred_elements) + " 个 float32（仅可见元素，同一存储位置只传一次）");
    row("来源存储块", fmt(h.storage_count) + " 块（每块在后段按原长分配一次）");
    row("位模式", "little-endian IEEE-754 binary32，base64 封装，保留 -0 / inf 等全部位模式");
    var button = $("download-handoff");
    button.onclick = function () { downloadPackage(h.package); };
  }

  // -- back segment run ---------------------------------------------------
  function onPackagePicked(event) {
    var file = event.target.files && event.target.files[0];
    clearBackErrors();
    if (!file) {
      selectedPackage = null;
      selectedPackageName = "";
      $("handoff-file-info").textContent = "尚未选择文件。原包可反复使用，换容量再试即可。";
      return;
    }
    var reader = new FileReader();
    reader.onload = function () {
      try {
        selectedPackage = JSON.parse(String(reader.result));
        selectedPackageName = file.name;
        var info = [];
        info.push(file.name + "（" + fmt(file.size) + " 字节）");
        if (selectedPackage && selectedPackage.split_op) {
          info.push("分界算子 " + selectedPackage.split_op.name);
        }
        if (selectedPackage && selectedPackage.transferred_elements !== undefined) {
          info.push("传输 " + fmt(selectedPackage.transferred_elements) + " 个元素");
        }
        $("handoff-file-info").textContent = info.join(" · ");
        setBackStatus("交接包已载入，可反复换容量试跑");
      } catch (err) {
        selectedPackage = null;
        $("handoff-file-info").textContent = file.name + " 解析失败：" + err.message;
        showBackErrors(["交接包不是合法 JSON：" + err.message]);
      }
    };
    reader.onerror = function () {
      selectedPackage = null;
      showBackErrors(["读取文件失败"]);
    };
    reader.readAsText(file, "utf-8");
  }

  async function runBack() {
    clearBackErrors();
    if (!selectedPackage) {
      showBackErrors(["请先选择前段产出的交接包（.json）"]);
      return;
    }
    var capacity = $("back-capacity").value.trim();
    if (!capacity) {
      showBackErrors(["请填写后段缓冲容量（按 float32 元素计）"]);
      return;
    }
    var button = $("run-back");
    button.disabled = true;
    button.textContent = "后段试跑中…";
    setBackStatus("正在载入交接包并继续执行…");
    try {
      var form = new FormData();
      form.append("capacity", capacity);
      form.append(
        "package",
        new Blob([JSON.stringify(selectedPackage)], { type: "application/json" }),
        selectedPackageName || "handoff.json"
      );
      var response = await fetch("/api/handoff/run", { method: "POST", body: form });
      var data = await response.json().catch(function () { return null; });
      if (!response.ok || !data) {
        var messages =
          data && data.errors && data.errors.length
            ? data.errors
            : ["请求失败（HTTP " + response.status + "），请稍后重试"];
        showBackErrors(messages);
        setBackStatus("");
        return;
      }
      render(data);
      setBackStatus(data.ok ? "后段试跑完成" : "后段容量不足，可保留原包换容量再试");
    } catch (err) {
      showBackErrors(["无法连接到服务器：" + err.message]);
      setBackStatus("");
    } finally {
      button.disabled = false;
      button.textContent = "开始后段试跑";
    }
  }

  // -- render -------------------------------------------------------------
  function render(data) {
    $("empty").hidden = true;
    $("result").hidden = false;

    var meta = [];
    meta.push("容量 " + fmt(data.capacity) + " 元素");
    meta.push(data.steps.length + " 步");
    if (data.stats.compactions) meta.push(data.stats.compactions + " 次整理");
    if (data.handoff_meta) {
      meta.push("后段 · 交接自 " + data.handoff_meta.split_op);
    }
    if (data.handoff) meta.push("前段 · 已生成交接包");
    $("result-meta").textContent = meta.join(" · ");

    renderStatus(data);
    renderStats(data);
    renderPeak(data);
    renderFailure(data);
    renderHandoff(data);
    renderOutputs(data);
    renderLayout(data);
    renderTimeline(data);
    renderWarnings(data);
  }

  function renderStatus(data) {
    var box = $("status");
    box.className = "status " + (data.ok ? "ok" : "bad");
    box.innerHTML = "";
    box.appendChild(el("span", "badge", data.ok ? "成功" : "失败"));
    var text;
    if (data.ok) {
      text =
        "执行完成：峰值占用 " + fmt(data.peak.elements) + " / " + fmt(data.capacity) +
        " 元素（" + data.peak.percent + "%），结束时占用 " +
        fmt(data.final_occupancy) + " 个元素。";
      if (data.handoff_meta) {
        text = "后段" + text;
      } else if (data.handoff) {
        text = "前段" + text + " 分界状态已打包，可在下方下载交接包。";
      }
    } else {
      text = data.failure.message;
    }
    box.appendChild(el("span", null, text));
  }

  function stat(key, valueHtml) {
    var node = el("div", "stat");
    node.appendChild(el("div", "k", key));
    var v = el("div", "v");
    v.innerHTML = valueHtml;
    node.appendChild(v);
    return node;
  }

  function renderStats(data) {
    var box = $("stats");
    box.innerHTML = "";
    if (data.ok) {
      box.appendChild(stat("缓冲容量", fmt(data.capacity) + " <small>元素</small>"));
      box.appendChild(
        stat("占用峰值", fmt(data.peak.elements) + " <small>" + data.peak.percent + "%</small>")
      );
      box.appendChild(
        stat(
          "结束占用",
          fmt(data.final_occupancy) + " <small>" + data.final_percent + "%</small>"
        )
      );
      box.appendChild(stat("分配次数", fmt(data.stats.allocations)));
      box.appendChild(
        stat(
          "整理次数",
          fmt(data.stats.compactions) +
          " <small>" + fmt(data.stats.moved_elements) + " 元素搬移</small>"
        )
      );
      box.appendChild(
        stat(
          "空闲 / 最大连续",
          fmt(data.stats.free_total) + " <small>/ " + fmt(data.stats.largest_free) + "</small>"
        )
      );
    } else {
      var f = data.failure;
      box.appendChild(stat("缓冲容量", fmt(f.capacity) + " <small>元素</small>"));
      box.appendChild(stat("失败时占用", fmt(data.final_occupancy) + " <small>元素</small>"));
      box.appendChild(stat("空闲总量", fmt(f.free_total) + " <small>元素</small>"));
      box.appendChild(stat("最大连续空闲", fmt(f.largest_free) + " <small>元素</small>"));
      box.appendChild(stat("本次需要", fmt(f.needed) + " <small>元素</small>"));
      box.appendChild(stat("失败步骤", "第 " + f.step + " 步"));
    }
  }

  function renderPeak(data) {
    var used = data.ok ? data.peak.elements : data.final_occupancy;
    var percent = data.capacity ? Math.min(100, (used / data.capacity) * 100) : 0;
    $("peak-fill").style.width = percent + "%";
    $("peak-cap").textContent = fmt(data.capacity) + " 元素";
    $("peak-caption").textContent = data.ok
      ? "峰值 " + fmt(data.peak.elements) + " 元素（" + data.peak.percent + "%），出现在「" +
        data.peak.at + "」"
      : "失败前占用 " + fmt(data.final_occupancy) + " 元素";
  }

  function renderFailure(data) {
    var block = $("failure-block");
    if (data.ok) {
      block.hidden = true;
      return;
    }
    block.hidden = false;
    var f = data.failure;
    var box = $("failure");
    box.innerHTML = "";
    box.appendChild(el("div", "msg", f.message));

    function row(key, value) {
      var r = el("div", "row");
      r.appendChild(el("span", "k", key));
      var v = el("span", "v");
      v.innerHTML = value;
      r.appendChild(v);
      box.appendChild(r);
    }
    if (f.phase_label) {
      row("卡住阶段", "<code>" + f.phase_label + "</code>（" +
        (f.phase === "load" ? "来源存储载入，尚未执行后段算子" : "后段算子执行中") + "）");
    }
    var action = f.action || {};
    var label;
    if (action.kind === "input") {
      label = "装入输入 " + action.name;
    } else if (action.kind === "load") {
      label = "载入 " + action.name + "（来源存储原长 " + fmt(action.length) + " 元素）";
    } else {
      label = "执行 " + (action.name || "?") + " = " + (action.kind || "?") +
        "(" + (action.operands || []).join(", ") + ")";
    }
    row("失败位置", "<code>第 " + f.step + " 步</code> · " + label);
    if (action.line) row("算子行号", "<code>第 " + action.line + " 行</code>");
    row("所需空间", "<code>" + fmt(f.needed) + "</code> 个连续元素");
    row("当时空闲", "<code>" + fmt(f.free_total) + "</code> 个元素，最大连续 <code>" +
      fmt(f.largest_free) + "</code> 个");
    row("容量", "<code>" + fmt(f.capacity) + "</code> 个元素");
    row("仍在使用", "<code>" + (f.live.length ? f.live.join(", ") : "无") + "</code>");
    if (f.compactions) row("此前整理", "<code>" + f.compactions + "</code> 次");
  }

  function renderOutputs(data) {
    var box = $("outputs");
    box.innerHTML = "";
    if (!data.outputs.length) {
      box.appendChild(el("p", "missing", "没有指定输出。"));
      return;
    }
    data.outputs.forEach(function (output) {
      var card = el("div", "output");
      var head = el("div", "output-head");
      head.appendChild(el("span", "output-name", output.name));
      if (output.missing) {
        head.appendChild(el("span", "pill", "缺失"));
        card.appendChild(head);
        card.appendChild(el("p", "missing", "该输出未在执行过程中产生。"));
        box.appendChild(card);
        return;
      }
      head.appendChild(el("span", "pill", "长度 " + fmt(output.length)));
      head.appendChild(
        el("span", "pill" + (output.role === "view" ? " view" : ""),
          output.role === "view" ? "切片视图" : "独立存储")
      );
      if (output.slice_of) {
        head.appendChild(el("span", "pill", "来源 " + output.slice_of));
      }
      head.appendChild(
        el("span", "pill storage", "存储 #" + output.storage + " · 偏移 " + output.offset)
      );
      card.appendChild(head);

      var values = el("div", "values");
      var limit = 64;
      output.values.slice(0, limit).forEach(function (value) {
        var chip = el("span", "val" + (value.charAt(0) === "-" ? " neg" : ""), value);
        values.appendChild(chip);
      });
      if (output.values.length > limit) {
        values.appendChild(
          el("span", "more", "… 其余 " + fmt(output.values.length - limit) + " 个")
        );
      }
      card.appendChild(values);
      box.appendChild(card);
    });
  }

  function renderLayout(data) {
    var bar = $("layout");
    var legend = $("layout-legend");
    bar.innerHTML = "";
    legend.innerHTML = "";

    var entries = data.layout || [];
    var total = data.capacity || 1;
    var usedEntries = entries.filter(function (entry) { return entry.kind === "live"; });
    $("layout-meta").textContent =
      usedEntries.length + " 块在用存储 · " + (entries.length - usedEntries.length) + " 段空闲";

    var viewStorage = entries.filter(function (entry) {
      return entry.kind === "live" && entry.views && entry.views.length;
    });
    $("layout-note").textContent = viewStorage.length
      ? "白色细条表示切片视图在所属存储内部的相对位置：" +
        viewStorage.map(function (entry) {
          return entry.label + " → " + entry.views.map(function (view) {
            return view.name + "[" + view.rel_start + ":" +
              (view.rel_start + view.length) + "]";
          }).join("、");
        }).join("；")
      : "当前没有存活的切片视图。";

    entries.forEach(function (entry) {
      var seg = el("div", "seg" + (entry.kind === "free" ? " free" : ""));
      seg.style.flexGrow = String(Math.max(entry.length, 0.0001));
      seg.style.flexBasis = "0";
      if (entry.kind === "free") {
        seg.title = "空闲 · " + entry.length + " 元素 · 起始 " + entry.start;
        if (entry.length / total > 0.05) {
          seg.appendChild(el("span", "seg-label", entry.length));
        }
      } else {
        var color = colorFor(entry.color);
        seg.style.background = color;
        var views = entry.views || [];
        var label = entry.label;
        if (views.length) {
          label += " +" + views.map(function (v) { return v.name; }).join(",");
        }
        seg.title =
          entry.label + " · " + entry.length + " 元素 · 起始 " + entry.start +
          (views.length
            ? " · 视图 " + views.map(function (v) {
                return v.name + "[" + v.rel_start + ":" + (v.rel_start + v.length) + "]";
              }).join(" ")
            : "");
        if (entry.length / total > 0.075) {
          seg.appendChild(el("span", "seg-label", label));
        }
        views.forEach(function (view) {
          var mark = el("div", "view-mark");
          mark.style.left = (view.rel_start / entry.length) * 100 + "%";
          mark.style.width = (view.length / entry.length) * 100 + "%";
          seg.appendChild(mark);
        });
      }
      bar.appendChild(seg);
    });

    usedEntries.forEach(function (entry) {
      var item = el("span", "item");
      var sw = el("span", "sw");
      sw.style.background = colorFor(entry.color);
      item.appendChild(sw);
      var text =
        entry.label + "（" + entry.length + "）" +
        (entry.views && entry.views.length ? "，视图占用同块" : "");
      item.appendChild(el("span", null, text));
      legend.appendChild(item);
    });
    var freeItem = el("span", "item");
    freeItem.appendChild(el("span", "sw free"));
    freeItem.appendChild(el("span", null, "空闲（斜纹）"));
    legend.appendChild(freeItem);
  }

  function renderTimeline(data) {
    var list = $("timeline");
    list.innerHTML = "";
    data.steps.forEach(function (step) {
      var item = el("li", "step");
      item.appendChild(el("span", "step-index", String(step.index)));

      var main = el("div", "step-main");
      main.appendChild(el("div", "step-title", step.title));
      main.appendChild(el("div", "step-detail", step.detail));

      var events = el("div", "step-events");
      step.allocated.forEach(function (alloc) {
        events.appendChild(
          el("span", "ev alloc", "分配 " + alloc.name + " · " + alloc.length + " @ " + alloc.offset)
        );
      });
      if (step.view) {
        events.appendChild(
          el("span", "ev view",
            "视图 " + step.view.name + " · 共享存储 #" + step.view.storage +
            " @ " + step.view.offset)
        );
      }
      step.freed.forEach(function (free) {
        events.appendChild(
          el("span", "ev free",
            (free.storage_freed ? "回收 " : "解除引用 ") + free.name +
            " · 存储 #" + free.storage +
            (free.storage_freed ? " 释放" : " 保留"))
        );
      });
      if (step.compaction) {
        var c = step.compaction;
        events.appendChild(
          el("span", "ev compact",
            "整理：搬移 " + c.moved + " 块 / " + c.moved_elements +
            " 元素，最大连续空闲 " + c.before_largest + " → " + c.after_largest)
        );
      }
      if (events.children.length) main.appendChild(events);
      item.appendChild(main);

      var meterWrap = el("div", "step-meter");
      var meter = el("div", "meter");
      var fill = el("i");
      fill.style.width = Math.min(100, step.percent) + "%";
      meter.appendChild(fill);
      meterWrap.appendChild(meter);
      meterWrap.appendChild(
        el("div", "meter-label", step.occupancy + " / " + data.capacity)
      );
      item.appendChild(meterWrap);

      list.appendChild(item);
    });
  }

  function renderWarnings(data) {
    var block = $("warnings-block");
    var list = $("warnings");
    if (!data.warnings || !data.warnings.length) {
      block.hidden = true;
      return;
    }
    block.hidden = false;
    list.innerHTML = "";
    data.warnings.forEach(function (warning) {
      list.appendChild(el("li", null, warning));
    });
  }

  // -- wire up ------------------------------------------------------------
  var initialized = false;
  function init() {
    if (initialized) return;
    initialized = true;
    document.querySelectorAll("[data-preset]").forEach(function (button) {
      button.addEventListener("click", function () {
        loadPreset(button.getAttribute("data-preset"));
      });
    });
    $("run").addEventListener("click", run);
    $("ops").addEventListener("input", function () { syncSplitOptions(); });
    $("handoff-file").addEventListener("change", onPackagePicked);
    $("run-back").addEventListener("click", runBack);
    document.addEventListener("keydown", function (event) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        run();
      }
    });
    loadPreset("slice");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
