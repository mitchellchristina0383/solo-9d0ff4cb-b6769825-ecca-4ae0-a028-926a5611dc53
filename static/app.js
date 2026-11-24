/* Memory dry-run page logic: submit plan, render report. */
(function () {
  "use strict";

  var PRESETS = {
    fragmentation: {
      capacity: "24",
      inputs: "a: 1, -4, 3, 2\nsep: 5, 5\nb: -1, 6, 7, -2\nc: 2, -3, 4, 8, -5, 1",
      ops: "u = relu(a)\nv = relu(b)\nw = relu(c)\nz = relu(sep)",
      outputs: "u, v, w, z",
    },
    slice: {
      capacity: "24",
      inputs: "X: 1, -2, 3, -4\nA: 10, -20, 30, -40, 50, -60\nY: -1, -2, 6, 7, -8, 9",
      ops: "v = slice(A, 1, 4)\np = relu(X)\nq = relu(Y)",
      outputs: "v, p, q",
    },
    full: {
      capacity: "10",
      inputs: "a: 1, 2, 3, 4\nb: 10, 20, 30, 40",
      ops: "c = add(a, b)",
      outputs: "c",
    },
  };

  var PALETTE = [
    "#58d3c4", "#7c9cff", "#f2b544", "#ff8fab",
    "#9ae66e", "#c79bff", "#f4845f", "#6fd3f7",
  ];

  var $ = function (id) { return document.getElementById(id); };

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
    clearErrors();
    setStatus("");
  }

  function setStatus(text) {
    $("run-status").textContent = text;
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

  function payload() {
    return {
      capacity: $("capacity").value.trim(),
      inputs: $("inputs").value,
      ops: $("ops").value,
      outputs: $("outputs").value,
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
      setStatus("");
    } catch (err) {
      showErrors(["无法连接到服务器：" + err.message]);
      $("empty").hidden = false;
      $("result").hidden = true;
      setStatus("");
    } finally {
      setBusy(false);
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
    $("result-meta").textContent = meta.join(" · ");

    renderStatus(data);
    renderStats(data);
    renderPeak(data);
    renderFailure(data);
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
    var action = f.action || {};
    var label = action.kind === "input"
      ? "装入输入 " + action.name
      : "执行 " + (action.name || "?") + " = " + (action.kind || "?") +
        "(" + (action.operands || []).join(", ") + ")";
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
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-preset]").forEach(function (button) {
      button.addEventListener("click", function () {
        loadPreset(button.getAttribute("data-preset"));
      });
    });
    $("run").addEventListener("click", run);
    document.addEventListener("keydown", function (event) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        run();
      }
    });
    loadPreset("slice");
  });
})();
